"""
05_transfer_function.py
Builds the transfer function mapping FWI to daily burn probability.

Methodology:
- Hybrid NBAC+NFDB approach:
  * NBAC (National Burned Area Composite): satellite-derived fire perimeters
    at 30m, aggregated to FWI grid to identify which cells burned each year
  * NFDB (National Fire Database): fire ignition point records with dates,
    used to assign fire occurrence to specific days
  * Only NFDB ignition points confirmed by NBAC spatial footprint are used
- Logistic Regression with prevalence correction
  (Bayes theorem correction for true fire prevalence)
- AUC evaluation on balanced subsample

Reference data:
- NBAC 1972-2024: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nbac
- NFDB: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt

Outputs:
  data/outputs/models/transfer_function_model.pkl
  data/outputs/results/transfer_function_lookup.csv
  data/outputs/results/ERA5-Land_historical_burn_probability.nc
  data/outputs/results/ERA5-Land_historical_annual_burn_probability.nc
"""

import rioxarray as rxr
from pyproj import Transformer
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from shapely.geometry import box
import pickle
import yaml
import os
import glob

with open("config.yml") as f:
    config = yaml.safe_load(f)

loc      = config["location"]
FWI_DIR  = config["data"]["processed"]["fwi"]
NBAC_DIR = config["data"]["raw"]["nbac"]
NFDB_DIR = config["data"]["raw"]["nfdb"]
MDL_DIR  = config["data"]["outputs"]["models"]
RES_DIR  = config["data"]["outputs"]["results"]
FIG_DIR  = config["data"]["outputs"]["figures"]
for d in [MDL_DIR, RES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

HIST_YEARS = range(config["cmip6"]["historical_period"][0],
                   config["cmip6"]["historical_period"][1] + 1)


# ── Load and prepare NBAC ──────────────────────────────────────────────────────
def load_nbac():
    shp = glob.glob(os.path.join(NBAC_DIR, "*.shp"))[0]
    nbac = gpd.read_file(shp)
    nbac = nbac[(nbac["YEAR"] >= HIST_YEARS.start) &
                (nbac["YEAR"] <= HIST_YEARS.stop - 1) &
                (nbac["PRESCRIBED"] != "Y")].copy()
    # Compute centroid in native projected CRS (Canada Lambert) before reprojecting
    # to avoid geographic CRS centroid distortion for large BC fire polygons
    nbac["centroid_proj"] = nbac.geometry.centroid
    nbac = nbac.to_crs("EPSG:4326")
    nbac["centroid_geo"] = nbac["centroid_proj"].to_crs("EPSG:4326")
    nbac = nbac.cx[loc["lon_min"]:loc["lon_max"], loc["lat_min"]:loc["lat_max"]].copy()
    print(f"NBAC fires in region: {len(nbac)}")
    return nbac


# ── Load and prepare NFDB ──────────────────────────────────────────────────────
def load_nfdb():
    shp = glob.glob(os.path.join(NFDB_DIR, "*.shp"))[0]
    nfdb = gpd.read_file(shp)
    nfdb = nfdb[(nfdb["LATITUDE"]  >= loc["lat_min"]) & (nfdb["LATITUDE"]  <= loc["lat_max"]) &
                (nfdb["LONGITUDE"] >= loc["lon_min"]) & (nfdb["LONGITUDE"] <= loc["lon_max"]) &
                (nfdb["YEAR"]  >= HIST_YEARS.start)   & (nfdb["YEAR"]  <= HIST_YEARS.stop - 1) &
                (nfdb["MONTH"] >= 4) & (nfdb["MONTH"] <= 10)].copy()
    print(f"NFDB fires in region: {len(nfdb)}")
    return nfdb


# ── Build burn mask from VLCE2 land cover ──────────────────────────────────────
def build_burn_mask(fwi_lats, fwi_lons):
    """
    Build a boolean burn mask on the ERA5-Land grid using VLCE2 land cover.

    VLCE2 classes (relevant to central BC domain):
      20  = Water              -> non-burnable
      31  = Snow/Ice           -> non-burnable
      32  = Rock/Rubble        -> non-burnable
      33  = Exposed Land       -> burnable (sparse veg, can carry fire)
      50  = Shrubland          -> burnable
      80  = Wetland            -> non-burnable
      81  = Wetland-Treed      -> burnable (burns under severe drought)
      100 = Herbs              -> burnable
      210 = Treed (mixed)      -> burnable  [dominant class, 61% of domain]
      220 = Treed (broadleaf)  -> burnable
      230 = Treed (conifer)    -> burnable

    Approach: reproject domain bounds to EPSG:3978, clip raster, compute
    modal class per ERA5-Land grid cell (0.1 deg ~ 8km), apply burn class mask.

    Reference: Hermosilla et al. (2022) VLCE2 product documentation.
    """
    NON_BURNABLE = {20, 31, 32, 80}  # water, snow/ice, rock, wetland

    lc_file = glob.glob(os.path.join(config["data"]["raw"]["landcover"], "*.tif"))[0]

    # Reproject domain bounds from EPSG:4326 to EPSG:3978 (VLCE2 native CRS)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)
    x_min, y_min = transformer.transform(loc["lon_min"], loc["lat_min"])
    x_max, y_max = transformer.transform(loc["lon_max"], loc["lat_max"])

    # Clip raster to domain — lazy load, only reads the relevant tiles
    ds_lc = rxr.open_rasterio(lc_file, chunks=512)
    ds_clip = ds_lc.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
    arr = ds_clip.squeeze().values  # (y, x) in EPSG:3978

    # Build x/y coordinate arrays for the clipped raster
    xs = ds_clip.x.values
    ys = ds_clip.y.values

    # For each ERA5-Land grid cell, find modal VLCE2 class within that cell
    # ERA5-Land cells are ~0.1 deg (~8km); VLCE2 is 30m — many pixels per cell
    inv_transformer = Transformer.from_crs("EPSG:3978", "EPSG:4326", always_xy=True)
    dlat = abs(fwi_lats[1] - fwi_lats[0])
    dlon = abs(fwi_lons[1] - fwi_lons[0])

    burn_mask = np.ones((len(fwi_lats), len(fwi_lons)), dtype=bool)

    for i, lat in enumerate(fwi_lats):
        for j, lon in enumerate(fwi_lons):
            # Cell bounds in EPSG:4326
            lat0, lat1 = lat - dlat / 2, lat + dlat / 2
            lon0, lon1 = lon - dlon / 2, lon + dlon / 2

            # Reproject cell corners to EPSG:3978
            cx0, cy0 = transformer.transform(lon0, lat0)
            cx1, cy1 = transformer.transform(lon1, lat1)

            # Index into raster
            xi = np.where((xs >= min(cx0, cx1)) & (xs <= max(cx0, cx1)))[0]
            yi = np.where((ys >= min(cy0, cy1)) & (ys <= max(cy0, cy1)))[0]

            if len(xi) == 0 or len(yi) == 0:
                # Cell outside raster extent — keep as burnable (domain edge)
                continue

            cell_vals = arr[np.ix_(yi, xi)].flatten()
            cell_vals = cell_vals[cell_vals != 255]  # remove fill

            if len(cell_vals) == 0:
                continue

            # Modal class
            modal_class = int(np.bincount(cell_vals.astype(int)).argmax())
            if modal_class in NON_BURNABLE:
                burn_mask[i, j] = False

    n_burnable     = int(burn_mask.sum())
    n_nonburnable  = int((~burn_mask).sum())
    print(f"Burn mask: {n_burnable} burnable cells, "
          f"{n_nonburnable} non-burnable cells "
          f"({100*n_nonburnable/(n_burnable+n_nonburnable):.1f}% masked)")

    return burn_mask


def build_fire_lookups(nbac, nfdb, fwi_lats, fwi_lons, lat_res, lon_res):
    # Build FWI grid GeoDataFrame
    cells = []
    for i, lat in enumerate(fwi_lats):
        for j, lon in enumerate(fwi_lons):
            cells.append({"lat":lat,"lon":lon,"lat_idx":i,"lon_idx":j,
                          "geometry":box(lon-lon_res/2, lat-lat_res/2,
                                         lon+lon_res/2, lat+lat_res/2)})
    grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")

    # NBAC: spatial join using projected centroid (reprojected to geographic)
    nbac_c = nbac.copy()
    nbac_c["geometry"] = nbac["centroid_geo"]
    joined = gpd.sjoin(nbac_c[["YEAR","geometry"]], grid_gdf, how="left", predicate="within")
    fire_cell_years = set()
    for _, row in joined.dropna().iterrows():
        fire_cell_years.add((int(row["YEAR"]), row["lat"], row["lon"]))
    print(f"NBAC unique fire cell-years: {len(fire_cell_years)}")

    # NFDB: ignition date + NBAC spatial confirmation
    def find_nearest(arr, v):
        return float(arr[np.abs(arr - v).argmin()])

    fire_lookup = set()
    for _, fire in nfdb.iterrows():
        yr, mo, dy = int(fire["YEAR"]), int(fire["MONTH"]), int(fire["DAY"])
        nlat = find_nearest(fwi_lats, fire["LATITUDE"])
        nlon = find_nearest(fwi_lons, fire["LONGITUDE"])
        if (yr, nlat, nlon) in fire_cell_years:
            fire_lookup.add((yr, mo, dy, nlat, nlon))
    print(f"Hybrid fire lookup: {len(fire_lookup)} unique day-cell combinations")

    return fire_cell_years, fire_lookup


# ── Build training dataset ─────────────────────────────────────────────────────
def build_training_data(ds_fwi, fire_lookup, fwi_lats, fwi_lons, burn_mask):
    """
    Build training dataset applying VLCE2 burn mask to exclude non-burnable cells.
    Non-burnable cells (water, rock, snow, wetland) are structurally impossible
    to burn and dilute the fire signal if included in training.
    """
    print("Building training dataset (vectorised)...")
    fwi_arr  = ds_fwi["fwi"].values  # (lat, lon, time)
    fire_arr = np.zeros_like(fwi_arr, dtype=np.uint8)

    time_vals = ds_fwi.time.values.astype("datetime64[D]")
    for (yr, mo, dy, lat_v, lon_v) in fire_lookup:
        li = np.argmin(np.abs(fwi_lats - lat_v))
        lj = np.argmin(np.abs(fwi_lons - lon_v))
        td = np.datetime64(f"{yr:04d}-{mo:02d}-{dy:02d}")
        tm = np.where(time_vals == td)[0]
        if len(tm) > 0:
            fire_arr[li, lj, tm[0]] = 1

    # Apply burn mask — set non-burnable cells to NaN in FWI
    # burn_mask is (lat, lon); fwi_arr is (lat, lon, time)
    fwi_masked = fwi_arr.copy()
    fwi_masked[~burn_mask, :] = np.nan

    fwi_flat  = fwi_masked.flatten()
    fire_flat = fire_arr.flatten()
    valid     = ~np.isnan(fwi_flat)
    X = fwi_flat[valid].reshape(-1, 1)
    y = fire_flat[valid]
    print(f"Total samples: {len(y):,} | Fire: {y.sum():,} ({100*y.mean():.4f}%)")
    return X, y


# ── Fit transfer function ──────────────────────────────────────────────────────
def fit_transfer_function(X, y):
    np.random.seed(42)
    fire_idx    = np.where(y == 1)[0]
    nonfire_idx = np.where(y == 0)[0]
    sampled     = np.random.choice(nonfire_idx, size=len(fire_idx)*50, replace=False)
    idx_bal     = np.concatenate([fire_idx, sampled])
    X_bal, y_bal = X[idx_bal], y[idx_bal]

    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X, y)

    auc = cross_val_score(lr, X_bal, y_bal, cv=5, scoring="roc_auc").mean()
    print(f"AUC: {auc:.3f}")

    true_prev  = y.mean()
    model_prev = lr.predict_proba(X)[:, 1].mean()
    cf         = true_prev / model_prev
    print(f"True prevalence: {100*true_prev:.4f}% | Correction factor: {cf:.6f}")

    return lr, true_prev, model_prev, cf, auc


# ── Apply transfer function ────────────────────────────────────────────────────
def apply_transfer_function(ds_fwi, lr, cf, burn_mask):
    """Apply transfer function; non-burnable cells get burn probability = 0."""
    fwi_vals = ds_fwi["fwi"].values
    bp       = np.zeros_like(fwi_vals, dtype=np.float32)

    # Only compute for burnable, non-NaN cells
    burnable_3d = np.broadcast_to(burn_mask[:, :, np.newaxis], fwi_vals.shape)
    valid = burnable_3d & ~np.isnan(fwi_vals)
    bp[valid] = lr.predict_proba(fwi_vals[valid].reshape(-1, 1))[:, 1] * cf

    # NaN where FWI is NaN regardless of burn mask
    bp[np.isnan(fwi_vals)] = np.nan

    return xr.DataArray(bp, coords=ds_fwi["fwi"].coords, dims=ds_fwi["fwi"].dims,
                        name="burn_probability")


def compute_annual_burn_prob(bp_da, years):
    annual = []
    for year in years:
        bp_yr = bp_da.sel(time=bp_da.time.dt.year == year)
        annual.append(1 - (1 - bp_yr).prod(dim="time"))
    ds = xr.concat(annual, dim="year")
    ds["year"] = list(years)
    return ds


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_transfer_function(X, y, lr, cf, fig_dir):
    fwi_range = np.linspace(0, 60, 300).reshape(-1,1)
    bp_corr   = lr.predict_proba(fwi_range)[:,1] * cf

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(fwi_range, bp_corr*1e4, color="red", linewidth=2)
    axes[0].set_xlabel("FWI"); axes[0].set_ylabel("Daily Burn Probability (×10⁻⁴)")
    axes[0].set_title("Transfer Function: FWI → Daily Burn Probability")
    axes[0].grid(True, alpha=0.3)

    fwi_fire    = X[y==1].flatten()
    fwi_nonfire = np.random.choice(X[y==0].flatten(), size=50000, replace=False)
    axes[1].hist(fwi_nonfire, bins=60, alpha=0.5, color="blue", label="Non-fire days", density=True)
    axes[1].hist(fwi_fire,    bins=30, alpha=0.5, color="red",  label="Fire ignition days", density=True)
    axes[1].set_xlabel("FWI"); axes[1].set_ylabel("Density")
    axes[1].set_title("FWI Distribution: Fire vs Non-fire Days"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Transfer Function: NBAC+NFDB Hybrid", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "transfer_function.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_annual_burn_prob(annual_bp, fig_dir):
    mean_bp = annual_bp.mean("year")
    ts      = annual_bp.mean(["lat","lon"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].contourf(mean_bp.lon, mean_bp.lat, mean_bp.values, levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=axes[0], label="Mean Annual Burn Probability")
    axes[0].set_title("Mean Annual Burn Probability (1980-2014)")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    axes[1].plot(range(1980,2015), ts.values, color="red", linewidth=1.5)
    axes[1].axhline(float(ts.mean()), color="black", linestyle="--",
                    label=f"Mean: {float(ts.mean()):.4f}")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Mean Annual Burn Probability")
    axes[1].set_title("Annual Burn Probability Time Series"); axes[1].legend()

    plt.suptitle("Annual Burn Probability: Central BC (1980-2014)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "annual_burn_probability.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load FWI
    ds_fwi = xr.open_dataset(os.path.join(FWI_DIR, "ERA5-Land_historical_fwi.nc"))
    print(f"FWI loaded: {ds_fwi.sizes}")
    fwi_lats = ds_fwi.lat.values
    fwi_lons = ds_fwi.lon.values
    lat_res  = abs(fwi_lats[1] - fwi_lats[0])
    lon_res  = abs(fwi_lons[1] - fwi_lons[0])

    # Load fire data
    nbac = load_nbac()
    nfdb = load_nfdb()

    # Build lookups
    _, fire_lookup = build_fire_lookups(nbac, nfdb, fwi_lats, fwi_lons, lat_res, lon_res)

    # Build burn mask from VLCE2
    print("Building burn mask from VLCE2 land cover...")
    burn_mask = build_burn_mask(fwi_lats, fwi_lons)

    # Build training data (burn mask excludes non-burnable cells)
    X, y = build_training_data(ds_fwi, fire_lookup, fwi_lats, fwi_lons, burn_mask)

    # Fit model
    lr, true_prev, model_prev, cf, auc = fit_transfer_function(X, y)

    # Save model — include burn_mask so script 06 can apply it consistently
    mdl_path = os.path.join(MDL_DIR, "transfer_function_model.pkl")
    with open(mdl_path, "wb") as f:
        pickle.dump({"model": lr, "true_prevalence": true_prev,
                     "model_prevalence": model_prev, "correction_factor": cf,
                     "burn_mask": burn_mask,
                     "fwi_lats": fwi_lats, "fwi_lons": fwi_lons}, f)
    print(f"Model saved: {mdl_path}")

    # Save lookup table
    fwi_range = np.linspace(0, 60, 300).reshape(-1, 1)
    bp_corr   = lr.predict_proba(fwi_range)[:, 1] * cf
    pd.DataFrame({"fwi": fwi_range.flatten(), "burn_probability": bp_corr}
                 ).to_csv(os.path.join(RES_DIR, "transfer_function_lookup.csv"), index=False)

    # Apply to ERA5-Land FWI
    bp_da     = apply_transfer_function(ds_fwi, lr, cf, burn_mask)
    annual_bp = compute_annual_burn_prob(bp_da, HIST_YEARS)
    print(f"Mean annual burn probability: {float(annual_bp.mean()):.4f}")
    print(f"Max annual burn probability:  {float(annual_bp.max()):.4f}")

    # Save
    for path, ds, name in [
        (os.path.join(RES_DIR, "ERA5-Land_historical_burn_probability.nc"),
         bp_da.to_dataset(name="burn_probability"), "burn_probability"),
        (os.path.join(RES_DIR, "ERA5-Land_historical_annual_burn_probability.nc"),
         annual_bp.to_dataset(name="annual_burn_probability"), "annual_burn_probability"),
    ]:
        if os.path.exists(path): os.remove(path)
        ds.to_netcdf(path)
    print("Burn probability outputs saved")

    # Plots
    plot_transfer_function(X, y, lr, cf, FIG_DIR)
    plot_annual_burn_prob(annual_bp, FIG_DIR)

    print(f"\nTransfer function summary:")
    print(f"  AUC: {auc:.3f}")
    print(f"  True fire prevalence: {100*true_prev:.4f}%")
    print(f"  Correction factor: {cf:.6f}")
    print("Done")

"""
04_fwi_computation.py
Computes Canadian Forest Fire Weather Index (FWI) system components.

Computes FWI for:
1. ERA5-Land (ground truth reference)
2. Bias-corrected CMIP6 historical (validation)
3. Bias-corrected CMIP6 future scenarios (ssp245, ssp585) for all models

Methodology:
- Van Wagner (1987) FWI system implemented via xclim
- Computed year-by-year to reinitialise carry-over indices each fire season
- Wind converted from m/s to km/h before computation
- Solar noon values used as inputs (extracted in script 01)

Key note: For future scenarios, the same bias correction parameters from the
historical period are applied forward (delta mapping approach).

Outputs: data/processed/fwi/
  ERA5-Land_historical_fwi.nc
  ERA5-Land_historical_fwi_components.nc
  {model}_{scenario}_fwi.nc
  {model}_{scenario}_fwi_components.nc
"""

import xarray as xr
import xclim
import xsdba
from xsdba import adjustment, processing
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

with open("config.yml") as f:
    config = yaml.safe_load(f)

ERA5_DIR  = config["data"]["processed"]["era5_land"]
CMIP_DIR  = config["data"]["raw"]["cmip6"]
BC_DIR    = config["data"]["processed"]["cmip6_bias_corrected"]
OUT_DIR   = config["data"]["processed"]["fwi"]
FIG_DIR   = config["data"]["outputs"]["figures"]
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS    = config["cmip6"]["models"]
SCENARIOS = ["historical"] + config["cmip6"]["scenarios"]
HIST_YEARS   = range(config["cmip6"]["historical_period"][0],
                     config["cmip6"]["historical_period"][1] + 1)
FUTURE_YEARS = range(config["cmip6"]["future_period"][0],
                     config["cmip6"]["future_period"][1] + 1)


# ── ERA5-Land prep ─────────────────────────────────────────────────────────────
def load_era5_land():
    files = sorted([os.path.join(ERA5_DIR, f"era5_land_{y}.nc") for y in HIST_YEARS])
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
    for v in ["t2m","d2m","u10","v10"]:
        ds[v] = ds[v].astype("float32")
    ds["u10"].attrs.update({"standard_name":"eastward_wind","units":"m s-1"})
    ds["v10"].attrs.update({"standard_name":"northward_wind","units":"m s-1"})
    ds["t2m"].attrs.update({"standard_name":"air_temperature","units":"K"})
    ds["d2m"].attrs.update({"standard_name":"dew_point_temperature","units":"K"})
    ds["sfcWind"] = xclim.atmos.wind_speed_from_vector(uas=ds["u10"], vas=ds["v10"])[0]
    ds["hurs"]    = xclim.atmos.relative_humidity_from_dewpoint(tas=ds["t2m"], tdps=ds["d2m"])
    ds["hurs"]    = ds["hurs"].chunk({"time":-1}).interpolate_na(dim="time")
    ds = ds[["t2m","hurs","sfcWind","tp"]].rename({"t2m":"tasmax","tp":"pr"})
    ds["pr"] = ds["pr"] * 1000
    ds["tasmax"].attrs["units"]  = "K"
    ds["hurs"].attrs["units"]    = "%"
    ds["sfcWind"].attrs["units"] = "m s-1"
    ds["pr"].attrs["units"]      = "mm day-1"
    ds["lat"].attrs["units"]     = "degrees_north"
    return ds


# ── FWI computation ────────────────────────────────────────────────────────────
def compute_fwi_yearly(ds, years):
    """Compute FWI year-by-year, reinitialising carry-over indices each season."""
    components = {k: [] for k in ["dc","dmc","ffmc","isi","bui","fwi"]}
    ds_wind = ds.copy()
    ds_wind["sfcWind_kmh"] = ds["sfcWind"] * 3.6
    ds_wind["sfcWind_kmh"].attrs["units"] = "km h-1"

    for year in years:
        ds_yr = ds_wind.sel(time=ds_wind.time.dt.year == year).chunk({"time":-1})
        if ds_yr.sizes["time"] == 0:
            continue
        with xclim.set_options(data_validation="log"):
            dc, dmc, ffmc, isi, bui, fwi = xclim.indices.cffwis_indices(
                tas=ds_yr["tasmax"], pr=ds_yr["pr"], hurs=ds_yr["hurs"],
                sfcWind=ds_yr["sfcWind_kmh"], lat=ds["lat"],
                season_method=None, overwintering=False)
        for k, v in zip(["dc","dmc","ffmc","isi","bui","fwi"],
                        [dc,dmc,ffmc,isi,bui,fwi]):
            components[k].append(v.compute())
        nan_frac = float(fwi.isnull().mean().compute())
        mean_fwi = float(fwi.mean().compute())
        print(f"  {year}: mean FWI={mean_fwi:.2f}, NaN={nan_frac:.3f}")

    return {k: xr.concat(v, dim="time") for k, v in components.items()}


# ── Apply bias correction to future scenarios ──────────────────────────────────
def load_and_bias_correct_future(model, scenario, ds_era5_rg):
    """
    Apply MBCn bias correction from the historical period to future scenarios.

    MBCn's adjust step requires sim and ref to have matching time lengths — it
    reorders sim to match the rank structure of ref. Passing the full 86-year
    future (18,404 fire-season days) against a 35-year ref (7,490 days) causes
    MBCn to exhaust the reference and collapse to the historical mean from 2050
    onwards.

    Fix: split the future into sequential 35-year windows (Cannon 2018 recommends
    this explicitly). Each window is adjusted independently against the same
    historical ref/hist, then concatenated. The last window (2085-2100, 16 years)
    is short but has sufficient samples (~3,400 days per cell) for nquantiles=50.
    Window-boundary artefacts are invisible after annual aggregation in scripts 05/06.

    Windows: 2015-2049 | 2050-2084 | 2085-2100
    """
    WINDOW_SIZE = 35  # years — must match historical training period length
    future_years = list(FUTURE_YEARS)

    # Build non-overlapping windows of WINDOW_SIZE years
    windows = []
    start = 0
    while start < len(future_years):
        windows.append(future_years[start: start + WINDOW_SIZE])
        start += WINDOW_SIZE
    print(f"  Future split into {len(windows)} windows: "
          + ", ".join(f"{w[0]}-{w[-1]}" for w in windows))

    # ── Load historical CMIP6 (training period — same for all windows) ─────────
    def load_cmip6_years(yr_range, scen):
        datasets = {}
        for var in ["tasmax", "hurs", "sfcWind", "pr"]:
            files = sorted([os.path.join(CMIP_DIR, f"{model}_{scen}_{var}_{y}.nc")
                            for y in yr_range])
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                print(f"  WARNING missing: {missing[:3]}...")
            present = [f for f in files if os.path.exists(f)]
            datasets[var] = xr.open_mfdataset(present, combine="by_coords")[var]
        ds = xr.Dataset(datasets)
        ds = ds.assign_coords(lon=(ds.lon.values - 360))
        ds["pr"] = ds["pr"] * 86400  # kg/m2/s -> mm/day
        unit_map = {"tasmax": "K", "hurs": "%", "sfcWind": "m s-1", "pr": "mm day-1"}
        for v in unit_map:
            ds[v].attrs["units"] = unit_map[v]
        return ds

    ds_hist = load_cmip6_years(HIST_YEARS, "historical")

    # Align ERA5 time to date-only (drop sub-daily component)
    ds_era5_rg2 = ds_era5_rg.chunk({"time": -1}).copy()
    ds_era5_rg2["time"] = ds_era5_rg2.time.values.astype("datetime64[D]").astype("datetime64[ns]")

    # Align historical CMIP6 time to ERA5 (both cover same 35-year fire-season days)
    ds_hist = ds_hist.chunk({"time": -1})
    ds_hist["time"] = ds_era5_rg2.time.values

    unit_map = {"tasmax": "K", "hurs": "%", "sfcWind": "m s-1", "pr": "mm day-1"}
    for v in unit_map:
        ds_era5_rg2[v].attrs["units"] = unit_map[v]
        ds_hist[v].attrs["units"]     = unit_map[v]

    # ── Train LOCI and MBCn once on the historical period ─────────────────────
    LOCI = adjustment.LOCI.train(
        ref=ds_era5_rg2["pr"], hist=ds_hist["pr"],
        group="time", thresh="0.1 mm day-1")

    ds_hist_loci = ds_hist.copy()
    ds_hist_loci["pr"] = LOCI.adjust(ds_hist["pr"])

    ref_st  = processing.stack_variables(ds_era5_rg2).compute()
    hist_st = processing.stack_variables(ds_hist_loci).compute()

    ADJ = xsdba.MBCn.train(
        ref=ref_st, hist=hist_st,
        base_kws={"nquantiles": 50, "group": "time"},
        adj_kws={"interp": "linear", "extrapolation": "constant"},
        n_iter=20, n_escore=-1)

    # ── Apply to each future window independently ──────────────────────────────
    corrected_windows = []

    for i, window_years in enumerate(windows):
        print(f"  Window {i+1}/{len(windows)}: {window_years[0]}-{window_years[-1]} "
              f"({len(window_years)} years)...")

        ds_win = load_cmip6_years(window_years, scenario)
        ds_win = ds_win.chunk({"time": -1})

        # LOCI precipitation correction
        ds_win_loci = ds_win.copy()
        ds_win_loci["pr"] = LOCI.adjust(ds_win["pr"])

        # MBCn: sim must match ref length — select matching-length ref/hist slice
        n_win = ds_win_loci.sizes["time"]
        n_ref = ref_st.sizes["time"]

        if n_win <= n_ref:
            # Window is shorter than or equal to training period — use first n_win
            # time steps of ref/hist so dimensions match
            ref_slice  = ref_st.isel(time=slice(0, n_win))
            hist_slice = hist_st.isel(time=slice(0, n_win))
        else:
            # Should not happen given WINDOW_SIZE == len(HIST_YEARS), but guard anyway
            raise ValueError(
                f"Window {window_years[0]}-{window_years[-1]} has {n_win} time steps "
                f"but ref has only {n_ref}. Reduce WINDOW_SIZE or check data.")

        sim_st = processing.stack_variables(ds_win_loci).compute()

        result = ADJ.adjust(sim=sim_st, ref=ref_slice, hist=hist_slice)
        result.attrs["units"] = ""

        ds_win_bc = xsdba.unstack_variables(result)
        ds_win_bc["hurs"] = ds_win_bc["hurs"].clip(0, 100)

        # Restore original calendar time coordinates from the raw future window
        ds_win_bc["time"] = ds_win["time"].values
        corrected_windows.append(ds_win_bc)

        mean_fwi_proxy = float(ds_win_bc["tasmax"].mean().compute())
        print(f"    mean tasmax after BC: {mean_fwi_proxy:.2f} K")

    # ── Concatenate all windows ────────────────────────────────────────────────
    ds_fut_bc = xr.concat(corrected_windows, dim="time")
    ds_fut_bc["hurs"] = ds_fut_bc["hurs"].clip(0, 100)

    print(f"  Future bias correction complete: {ds_fut_bc.sizes['time']} time steps total")
    return ds_fut_bc


# ── Regrid ERA5 to CMIP6 ───────────────────────────────────────────────────────
def add_bounds(ds):
    lat, lon = ds.lat.values, ds.lon.values
    dlat = abs(lat[1]-lat[0]); dlon = abs(lon[1]-lon[0])
    lat_b = np.concatenate([[lat[0]+dlat/2*np.sign(lat[0]-lat[1])],
                             (lat[:-1]+lat[1:])/2,
                             [lat[-1]-dlat/2*np.sign(lat[0]-lat[1])]])
    lon_b = np.concatenate([[lon[0]-dlon/2],(lon[:-1]+lon[1:])/2,[lon[-1]+dlon/2]])
    return ds.assign_coords(lat_b=("lat_b",lat_b), lon_b=("lon_b",lon_b))


def get_cmip6_grid(model):
    """Load one CMIP6 file to get the grid."""
    f = os.path.join(CMIP_DIR, f"{model}_historical_tasmax_1980.nc")
    ds = xr.open_dataset(f)
    ds = ds.assign_coords(lon=(ds.lon.values - 360))
    return ds


def regrid_era5_to_cmip6(ds_era5, ds_cmip6):
    e5b = add_bounds(ds_era5)
    c6b = add_bounds(ds_cmip6)
    bl  = xe.Regridder(e5b[["tasmax","hurs","sfcWind"]], c6b, method="bilinear")
    con = xe.Regridder(e5b[["pr"]], c6b, method="conservative")
    return xr.merge([bl(e5b[["tasmax","hurs","sfcWind"]]), con(e5b[["pr"]])])


# ── Save helper ────────────────────────────────────────────────────────────────
def save_fwi(components, tag):
    fwi_path = os.path.join(OUT_DIR, f"{tag}_fwi.nc")
    comp_path = os.path.join(OUT_DIR, f"{tag}_fwi_components.nc")
    for p in [fwi_path, comp_path]:
        if os.path.exists(p): os.remove(p)
    components["fwi"].to_dataset(name="fwi").to_netcdf(fwi_path)
    xr.Dataset({k: components[k] for k in ["dc","dmc","ffmc","isi","bui","fwi"]}
               ).to_netcdf(comp_path)
    nan_frac = float(components["fwi"].isnull().mean())
    mean_fwi = float(components["fwi"].mean())
    print(f"  Saved {tag}: mean FWI={mean_fwi:.2f}, NaN={nan_frac:.3f}")


# ── Validation plot ────────────────────────────────────────────────────────────
def plot_fwi_validation(fwi_era5, fwi_bc, model, fig_dir):
    fwi_era5_ann = fwi_era5.resample(time="YE").max().compute()
    fwi_bc_ann   = fwi_bc.resample(time="YE").max().compute()
    ts_era5 = fwi_era5_ann.mean(["lat","lon"])
    ts_bc   = fwi_bc_ann.mean(["lat","lon"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ts_era5.time.dt.year, ts_era5.values, color="blue", label="ERA5-Land")
    axes[0].plot(ts_bc.time.dt.year,   ts_bc.values,   color="green", linestyle="--", label="CMIP6 bias-corrected")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Mean Annual Max FWI")
    axes[0].set_title("Annual Maximum FWI"); axes[0].legend()

    e5f = fwi_era5.values.flatten(); e5f = e5f[~np.isnan(e5f)]
    bcf = fwi_bc.values.flatten();   bcf = bcf[~np.isnan(bcf)]
    axes[1].hist(e5f, bins=80, alpha=0.5, color="blue",  label="ERA5-Land", density=True)
    axes[1].hist(bcf, bins=80, alpha=0.5, color="green", label="CMIP6 bias-corrected", density=True)
    axes[1].set_xlabel("FWI"); axes[1].set_ylabel("Density"); axes[1].set_title("Daily FWI Distribution"); axes[1].legend()

    plt.suptitle(f"FWI Validation: {model}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{model}_historical_fwi_validation.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ERA5-Land FWI
    era5_fwi_path = os.path.join(OUT_DIR, "ERA5-Land_historical_fwi.nc")
    if not os.path.exists(era5_fwi_path):
        print("=== ERA5-Land FWI ===")
        ds_era5 = load_era5_land()
        comps   = compute_fwi_yearly(ds_era5, HIST_YEARS)
        save_fwi(comps, "ERA5-Land_historical")
    else:
        print("ERA5-Land FWI exists, skipping")

    # CMIP6 FWI — historical and future scenarios
    ds_era5 = load_era5_land()

    for model in MODELS:
        # Get CMIP6 grid for this model
        ds_cmip6_grid = get_cmip6_grid(model)
        ds_era5_rg = regrid_era5_to_cmip6(ds_era5, ds_cmip6_grid)
        ds_era5_rg["lat"].attrs["units"] = "degrees_north"

        # Historical
        hist_fwi_path = os.path.join(OUT_DIR, f"{model}_historical_fwi.nc")
        if not os.path.exists(hist_fwi_path):
            print(f"\n=== {model} historical FWI ===")
            bc_path = os.path.join(BC_DIR, f"{model}_historical_bias_corrected.nc")
            ds_bc = xr.open_dataset(bc_path)
            ds_bc["time"] = ds_bc.time.values.astype("datetime64[D]").astype("datetime64[ns]")
            ds_bc["lat"].attrs["units"] = "degrees_north"
            comps = compute_fwi_yearly(ds_bc, HIST_YEARS)
            save_fwi(comps, f"{model}_historical")
            plot_fwi_validation(
                xr.open_dataset(era5_fwi_path)["fwi"],
                comps["fwi"], model, FIG_DIR)
        else:
            print(f"{model} historical FWI exists, skipping")

        # Future scenarios
        for scenario in config["cmip6"]["scenarios"]:
            fut_fwi_path = os.path.join(OUT_DIR, f"{model}_{scenario}_fwi.nc")
            if not os.path.exists(fut_fwi_path):
                print(f"\n=== {model} {scenario} FWI ===")
                ds_bc_fut = load_and_bias_correct_future(model, scenario, ds_era5_rg)
                ds_bc_fut["lat"].attrs["units"] = "degrees_north"
                comps = compute_fwi_yearly(ds_bc_fut, FUTURE_YEARS)
                save_fwi(comps, f"{model}_{scenario}")
            else:
                print(f"{model} {scenario} FWI exists, skipping")

    print("\nFWI computation complete")

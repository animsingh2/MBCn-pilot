"""
03_bias_correction.py
Bias-corrects CMIP6 against ERA5-Land using LOCI + MBCn.

Methodology:
- LOCI (Local Intensity Scaling): corrects precipitation wet day frequency
  Reference: Themessl et al. (2012)
- MBCn (Multivariate Bias Correction): corrects all variables simultaneously,
  preserving intervariable co-occurrence structure (e.g. hot+dry days)
  Reference: Cannon (2018), validated on Canadian FWI
- Regridding: ERA5-Land (0.1 deg) -> CMIP6 grid (0.25 deg)
  Bilinear for temperature/humidity/wind, conservative for precipitation

Runs for: all models x historical scenario
Future scenarios (ssp245, ssp585) use same bias correction parameters
applied forward (handled in 04_fwi_computation.py)

Outputs: data/processed/cmip6_bias_corrected/{model}_historical_bias_corrected.nc
"""

import xarray as xr
import xsdba
from xsdba import adjustment, processing
import xesmf as xe
import xclim
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

with open("config.yml") as f:
    config = yaml.safe_load(f)

loc      = config["location"]
ERA5_DIR = config["data"]["processed"]["era5_land"]
CMIP_DIR = config["data"]["raw"]["cmip6"]
OUT_DIR  = config["data"]["processed"]["cmip6_bias_corrected"]
FIG_DIR  = config["data"]["outputs"]["figures"]
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS    = config["cmip6"]["models"]
VARIABLES = ["tasmax", "hurs", "sfcWind", "pr"]
HIST_YEARS = range(config["cmip6"]["historical_period"][0],
                   config["cmip6"]["historical_period"][1] + 1)


# ── Load ERA5-Land ─────────────────────────────────────────────────────────────
def load_era5_land():
    files = sorted([os.path.join(ERA5_DIR, f"era5_land_{y}.nc") for y in HIST_YEARS])
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})

    # Cast to float32
    for v in ["t2m","d2m","u10","v10"]:
        ds[v] = ds[v].astype("float32")

    # CF attributes for xclim
    ds["u10"].attrs.update({"standard_name":"eastward_wind","units":"m s-1"})
    ds["v10"].attrs.update({"standard_name":"northward_wind","units":"m s-1"})
    ds["t2m"].attrs.update({"standard_name":"air_temperature","units":"K"})
    ds["d2m"].attrs.update({"standard_name":"dew_point_temperature","units":"K"})

    # Wind speed and relative humidity
    ds["sfcWind"] = xclim.atmos.wind_speed_from_vector(uas=ds["u10"], vas=ds["v10"])[0]
    ds["hurs"]    = xclim.atmos.relative_humidity_from_dewpoint(tas=ds["t2m"], tdps=ds["d2m"])

    # Interpolate NaN humidity (Sonntag formula edge cases at near-saturation)
    ds["hurs"] = ds["hurs"].chunk({"time": -1}).interpolate_na(dim="time")

    ds = ds[["t2m","hurs","sfcWind","tp"]].rename({"t2m":"tasmax","tp":"pr"})
    ds["pr"] = ds["pr"] * 1000  # m -> mm/day

    ds["tasmax"].attrs["units"] = "K"
    ds["hurs"].attrs["units"]   = "%"
    ds["sfcWind"].attrs["units"]= "m s-1"
    ds["pr"].attrs["units"]     = "mm day-1"

    return ds


# ── Load CMIP6 ─────────────────────────────────────────────────────────────────
def load_cmip6(model, scenario, years):
    datasets = {}
    for var in VARIABLES:
        files = sorted([os.path.join(CMIP_DIR, f"{model}_{scenario}_{var}_{y}.nc") for y in years])
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"  WARNING missing: {missing}"); continue
        datasets[var] = xr.open_mfdataset(files, combine="by_coords")[var]
    ds = xr.Dataset(datasets)
    ds = ds.assign_coords(lon=(ds.lon.values - 360))
    time_idx = ds.indexes["time"]
    if hasattr(time_idx, "to_datetimeindex"):
        ds["time"] = time_idx.to_datetimeindex(time_unit="ns")
    ds["tasmax"].attrs["units"] = "K"
    ds["hurs"].attrs["units"]   = "%"
    ds["sfcWind"].attrs["units"]= "m s-1"
    ds["pr"].attrs["units"]     = "mm day-1"
    ds["pr"] = ds["pr"] * 86400  # kg/m2/s -> mm/day
    return ds


# ── Regrid ─────────────────────────────────────────────────────────────────────
def add_bounds(ds):
    lat, lon = ds.lat.values, ds.lon.values
    dlat = abs(lat[1]-lat[0])
    dlon = abs(lon[1]-lon[0])
    lat_b = np.concatenate([[lat[0]+dlat/2*np.sign(lat[0]-lat[1])],
                             (lat[:-1]+lat[1:])/2,
                             [lat[-1]-dlat/2*np.sign(lat[0]-lat[1])]])
    lon_b = np.concatenate([[lon[0]-dlon/2], (lon[:-1]+lon[1:])/2, [lon[-1]+dlon/2]])
    return ds.assign_coords(lat_b=("lat_b",lat_b), lon_b=("lon_b",lon_b))


def regrid_era5_to_cmip6(ds_era5, ds_cmip6):
    e5b = add_bounds(ds_era5)
    c6b = add_bounds(ds_cmip6)
    bl  = xe.Regridder(e5b[["tasmax","hurs","sfcWind"]], c6b, method="bilinear")
    con = xe.Regridder(e5b[["pr"]], c6b, method="conservative")
    return xr.merge([bl(e5b[["tasmax","hurs","sfcWind"]]), con(e5b[["pr"]])])


# ── Bias correction ────────────────────────────────────────────────────────────
def bias_correct(ds_era5_rg, ds_cmip6):
    # Align time coordinates (ERA5 00:00, CMIP6 12:00 -> both to date)
    ds_era5_rg = ds_era5_rg.chunk({"time":-1})
    ds_cmip6   = ds_cmip6.chunk({"time":-1})
    ds_era5_rg["time"] = ds_era5_rg.time.values.astype("datetime64[D]").astype("datetime64[ns]")
    ds_cmip6["time"]   = ds_era5_rg.time.values

    unit_map = {"tasmax":"K","hurs":"%","sfcWind":"m s-1","pr":"mm day-1"}
    for v in VARIABLES:
        ds_era5_rg[v].attrs["units"] = unit_map[v]
        ds_cmip6[v].attrs["units"]   = unit_map[v]

    # LOCI: correct precipitation wet day frequency
    LOCI = adjustment.LOCI.train(ref=ds_era5_rg["pr"], hist=ds_cmip6["pr"],
                                  group="time", thresh="0.1 mm day-1")
    ds_cmip6_bc = ds_cmip6.copy()
    ds_cmip6_bc["pr"] = LOCI.adjust(ds_cmip6["pr"])

    # MBCn: multivariate bias correction
    ref_st  = processing.stack_variables(ds_era5_rg).compute()
    hist_st = processing.stack_variables(ds_cmip6_bc).compute()

    ADJ = xsdba.MBCn.train(ref=ref_st, hist=hist_st,
                            base_kws={"nquantiles":50,"group":"time"},
                            adj_kws={"interp":"linear","extrapolation":"constant"},
                            n_iter=20, n_escore=-1)
    result = ADJ.adjust(sim=hist_st, ref=ref_st, hist=hist_st)
    result.attrs["units"] = ""
    ds_bc = xsdba.unstack_variables(result)
    ds_bc["hurs"] = ds_bc["hurs"].clip(0, 100)
    return ds_bc, LOCI, ADJ


# ── Validation plot ────────────────────────────────────────────────────────────
def plot_validation(ds_era5_rg, ds_cmip6, ds_bc, model, scenario, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    labels = {"tasmax":"Temperature (K)","hurs":"Relative Humidity (%)","sfcWind":"Wind Speed (m/s)","pr":"Precipitation (mm/day)"}
    for i, var in enumerate(["tasmax","hurs","sfcWind","pr"]):
        ax = axes[i]
        e5  = ds_era5_rg[var].values.flatten(); e5  = e5[~np.isnan(e5)]
        c6  = ds_cmip6[var].values.flatten();   c6  = c6[~np.isnan(c6)]
        bc  = ds_bc[var].values.flatten();       bc  = bc[~np.isnan(bc)]
        if var == "pr":
            e5, c6, bc = e5[e5>0.01], c6[c6>0.01], bc[bc>0.01]
            ax.set_xscale("log")
        ax.hist(e5, bins=80, alpha=0.5, color="blue",  label="ERA5-Land", density=True)
        ax.hist(c6, bins=80, alpha=0.5, color="red",   label="CMIP6 raw", density=True)
        ax.hist(bc, bins=80, alpha=0.5, color="green", label="CMIP6 bias-corrected", density=True)
        ax.set_xlabel(labels[var]); ax.set_ylabel("Density"); ax.set_title(var); ax.legend(fontsize=8)
    plt.suptitle(f"Bias Correction: {model} {scenario} vs ERA5-Land", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{model}_{scenario}_bias_correction_validation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading ERA5-Land...")
    ds_era5 = load_era5_land()
    print(f"ERA5-Land loaded: {ds_era5.sizes}")

    for model in MODELS:
        out_path = os.path.join(OUT_DIR, f"{model}_historical_bias_corrected.nc")
        if os.path.exists(out_path):
            print(f"{model}: bias-corrected file exists, skipping")
            continue

        print(f"\n=== {model} ===")
        ds_cmip6 = load_cmip6(model, "historical", HIST_YEARS)
        print(f"CMIP6 loaded: {ds_cmip6.sizes}")

        ds_era5_rg = regrid_era5_to_cmip6(ds_era5, ds_cmip6)
        print(f"ERA5-Land regridded to CMIP6 grid: {ds_era5_rg.sizes}")

        ds_bc, _, _ = bias_correct(ds_era5_rg, ds_cmip6)

        nan_count = int(ds_bc["tasmax"].isnull().sum().compute())
        bc_wet    = (ds_bc["pr"].values.flatten() > 0.1).mean() * 100
        print(f"NaNs in tasmax: {nan_count}")
        print(f"Bias-corrected wet day frequency: {bc_wet:.1f}%")

        plot_validation(ds_era5_rg, ds_cmip6, ds_bc, model, "historical", FIG_DIR)

        if os.path.exists(out_path):
            os.remove(out_path)
        ds_bc.to_netcdf(out_path)
        print(f"Saved: {out_path}")

    print("\nBias correction complete")

"""
02_download_cmip6.py
Downloads NEX-GDDP-CMIP6 data for all models, scenarios, and variables.
Uses AWS S3 public bucket (no credentials required).
Subsets to study bounding box on download to minimise storage.

Models:  CanESM5, MPI-ESM1-2-HR, ACCESS-CM2
Scenarios: historical (1980-2014), ssp245 (2015-2100), ssp585 (2015-2100)
Variables: tasmax, hurs, sfcWind, pr

Outputs: data/raw/cmip6/{model}_{scenario}_{variable}_{year}.nc
"""

import xarray as xr
import s3fs
import yaml
import os

with open("config.yml") as f:
    config = yaml.safe_load(f)

loc      = config["location"]
OUT_DIR  = config["data"]["raw"]["cmip6"]
os.makedirs(OUT_DIR, exist_ok=True)

MODELS    = config["cmip6"]["models"]
SCENARIOS = ["historical"] + config["cmip6"]["scenarios"]
VARIABLES = ["tasmax", "hurs", "sfcWind", "pr"]
ENSEMBLE  = "r1i1p1f1"
BASE_PATH = "nex-gddp-cmip6/NEX-GDDP-CMIP6"

YEAR_RANGES = {
    "historical": range(config["cmip6"]["historical_period"][0],
                        config["cmip6"]["historical_period"][1] + 1),
    "ssp245":     range(config["cmip6"]["future_period"][0],
                        config["cmip6"]["future_period"][1] + 1),
    "ssp585":     range(config["cmip6"]["future_period"][0],
                        config["cmip6"]["future_period"][1] + 1),
}

fs = s3fs.S3FileSystem(anon=True)


def download_file(model, scenario, variable, year):
    out = os.path.join(OUT_DIR, f"{model}_{scenario}_{variable}_{year}.nc")
    if os.path.exists(out):
        return

    fname = f"{variable}_day_{model}_{scenario}_{ENSEMBLE}_gn_{year}_v2.0.nc"
    s3    = f"{BASE_PATH}/{model}/{scenario}/{ENSEMBLE}/{variable}/{fname}"

    try:
        with fs.open(s3) as f:
            ds = xr.open_dataset(f, engine="h5netcdf")
            ds_sub = ds.sel(
                lat=slice(loc["lat_min"], loc["lat_max"]),
                lon=slice(loc["lon_min"] + 360, loc["lon_max"] + 360),
            )
            ds_sub = ds_sub.sel(
                time=ds_sub.time.dt.month.isin([4,5,6,7,8,9,10])
            )
            ds_sub.to_netcdf(out)
            print(f"  saved: {model} {scenario} {variable} {year}")
    except Exception as e:
        print(f"  FAILED {model} {scenario} {variable} {year}: {e}")


if __name__ == "__main__":
    for model in MODELS:
        for scenario in SCENARIOS:
            print(f"\n{model} {scenario}")
            for variable in VARIABLES:
                for year in YEAR_RANGES[scenario]:
                    download_file(model, scenario, variable, year)
    print("\nDownload complete")

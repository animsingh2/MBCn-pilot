"""
01_download_era5_land.py
Downloads hourly ERA5-Land and extracts solar noon values for FWI computation.

Methodology:
- Solar noon computed per grid cell using ephem (Duffie et al. 2020)
- Temperature, dewpoint, wind: interpolated between hourly values bracketing solar noon
- Precipitation: 23:00 UTC daily accumulation (ERA5-Land resets at midnight UTC)

Outputs: data/processed/era5_land/era5_land_{year}.nc  (1980-2014)
"""

import cdsapi, zipfile, shutil, ephem
import numpy as np
import pandas as pd
import xarray as xr
import yaml, os

with open("config.yml") as f:
    config = yaml.safe_load(f)

loc     = config["location"]
RAW_DIR = config["data"]["raw"]["era5_land_hourly"]
OUT_DIR = config["data"]["processed"]["era5_land"]
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

VARIABLES   = ["2m_temperature","2m_dewpoint_temperature",
               "10m_u_component_of_wind","10m_v_component_of_wind","total_precipitation"]
FIRE_MONTHS = ["04","05","06","07","08","09","10"]
ALL_HOURS   = [f"{h:02d}:00" for h in range(24)]
YEARS       = range(1980, 2015)


def get_solar_noon_utc(lons, date):
    """Solar noon UTC hour for each longitude (latitude effect on timing is negligible)."""
    sn = np.zeros(len(lons))
    for j, lon in enumerate(lons):
        obs = ephem.Observer()
        obs.lat, obs.lon = "52.5", str(lon)
        obs.date = date.strftime("%Y/%m/%d")
        t = ephem.Date(obs.next_transit(ephem.Sun())).datetime()
        sn[j] = t.hour + t.minute/60 + t.second/3600
    return sn


def extract_solar_noon(ds_hourly, lats, lons):
    """Extract solar noon slice for instantaneous vars; 23:00 UTC for precipitation."""
    dates = pd.DatetimeIndex(ds_hourly.valid_time.values).normalize().unique()
    daily = []
    for date in dates:
        ds_day   = ds_hourly.sel(valid_time=ds_hourly.valid_time.dt.date == date.date())
        sol_noon = get_solar_noon_utc(lons, date.date())
        dvars    = {}
        for var in ["t2m","d2m","u10","v10"]:
            arr = np.full((len(lats), len(lons)), np.nan)
            for j, (_, sn) in enumerate(zip(lons, sol_noon)):
                h0, h1   = int(sn), (int(sn)+1) % 24
                w1, w0   = sn - int(sn), 1 - (sn - int(sn))
                v0 = ds_day[var].sel(valid_time=ds_day.valid_time.dt.hour==h0).isel(valid_time=0).values[:,j]
                v1 = ds_day[var].sel(valid_time=ds_day.valid_time.dt.hour==h1).isel(valid_time=0).values[:,j]
                arr[:,j] = w0*v0 + w1*v1
            dvars[var] = xr.DataArray(arr[np.newaxis],
                coords={"valid_time":[date],"latitude":lats,"longitude":lons},
                dims=["valid_time","latitude","longitude"])
        tp = ds_day["tp"].sel(valid_time=ds_day.valid_time.dt.hour==23).isel(valid_time=0).values
        dvars["tp"] = xr.DataArray(tp[np.newaxis],
            coords={"valid_time":[date],"latitude":lats,"longitude":lons},
            dims=["valid_time","latitude","longitude"])
        daily.append(xr.Dataset(dvars))
    return xr.concat(daily, dim="valid_time")


def download_month(c, year, month):
    out = os.path.join(RAW_DIR, f"era5_land_{year}_{month}.nc")
    if os.path.exists(out):
        print(f"  {year}-{month}: exists, skipping"); return
    print(f"  Downloading {year}-{month}...")
    c.retrieve("reanalysis-era5-land", {
        "variable": VARIABLES, "year": str(year), "month": month,
        "day": [f"{d:02d}" for d in range(1,32)], "time": ALL_HOURS,
        "area": [loc["lat_max"],loc["lon_min"],loc["lat_min"],loc["lon_max"]],
        "format": "netcdf"}, out)
    print(f"  {year}-{month}: done")


def process_year(year):
    out = os.path.join(OUT_DIR, f"era5_land_{year}.nc")
    if os.path.exists(out):
        print(f"{year}: exists, skipping"); return
    print(f"Processing {year}...")
    datasets = []
    for month in FIRE_MONTHS:
        zp = os.path.join(RAW_DIR, f"era5_land_{year}_{month}.nc")
        if not os.path.exists(zp):
            print(f"  WARNING: missing {zp}"); continue
        ext = os.path.join(RAW_DIR, f"tmp_{year}_{month}")
        os.makedirs(ext, exist_ok=True)
        with zipfile.ZipFile(zp,"r") as z: z.extractall(ext)
        ds = xr.open_dataset(os.path.join(ext,"data_0.nc"), engine="netcdf4")
        datasets.append(extract_solar_noon(ds, ds.latitude.values, ds.longitude.values))
        ds.close(); shutil.rmtree(ext)
        print(f"  {month}: done")
    if datasets:
        ds_yr = xr.concat(datasets, dim="valid_time")
        ds_yr.to_netcdf(out)
        print(f"{year}: saved ({ds_yr.sizes['valid_time']} days)")


if __name__ == "__main__":
    print("=== Downloading ERA5-Land ===")
    c = cdsapi.Client()
    for year in YEARS:
        for month in FIRE_MONTHS:
            download_month(c, year, month)

    print("\n=== Extracting solar noon values ===")
    for year in YEARS:
        process_year(year)
    print("Done")

"""
06_return_periods.py
Computes annual burn probability and return period curves for all models and scenarios.

Methodology:
- Applies transfer function to CMIP6 FWI outputs for all models x scenarios
- Annual burn probability: P(fire in year) = 1 - prod(1 - P_daily)
- Return period curves fit using GEV (Generalised Extreme Value) distribution
- Multi-model ensemble: mean and spread across 3 CMIP6 models
- Climate change signal: future (2015-2100) vs historical (1980-2014)

Outputs:
  data/outputs/results/
    {model}_{scenario}_annual_burn_probability.nc   (all models x scenarios)
    return_period_table.csv                          (ERA5-Land + all ensemble members)
    ensemble_return_period_table.csv                 (ensemble mean + 5th/95th percentile)
  data/outputs/figures/
    return_period_analysis.png
    climate_change_signal.png
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yaml
import os
from scipy import stats

with open("config.yml") as f:
    config = yaml.safe_load(f)

FWI_DIR = config["data"]["processed"]["fwi"]
RES_DIR = config["data"]["outputs"]["results"]
FIG_DIR = config["data"]["outputs"]["figures"]
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS       = config["cmip6"]["models"]
SCENARIOS    = config["cmip6"]["scenarios"]
HIST_YEARS   = range(config["cmip6"]["historical_period"][0],
                     config["cmip6"]["historical_period"][1] + 1)
FUTURE_YEARS = range(config["cmip6"]["future_period"][0],
                     config["cmip6"]["future_period"][1] + 1)


# ── Load transfer function ─────────────────────────────────────────────────────
with open(os.path.join(config["data"]["outputs"]["models"],
                       "transfer_function_model.pkl"), "rb") as f:
    tf = pickle.load(f)

lr_model  = tf["model"]
cf        = tf["correction_factor"]
burn_mask = tf["burn_mask"]  # (lat, lon) boolean array from VLCE2


# ── Apply transfer function to FWI ────────────────────────────────────────────
def get_burn_mask_for_grid(fwi_lats, fwi_lons):
    """
    Reproject burn mask to match the FWI grid using nearest-neighbour lookup.
    ERA5-Land burn mask is 31x51 (0.1 deg); CMIP6 FWI is 12x20 (0.25 deg).
    """
    bm_lats = tf["fwi_lats"]  # ERA5-Land lat coords the mask was built on
    bm_lons = tf["fwi_lons"]

    if (len(fwi_lats) == len(bm_lats) and
            np.allclose(fwi_lats, bm_lats) and
            np.allclose(fwi_lons, bm_lons)):
        return burn_mask  # same grid, no regridding needed

    # Nearest-neighbour: for each target cell find closest source cell
    mask_regrid = np.ones((len(fwi_lats), len(fwi_lons)), dtype=bool)
    for i, lat in enumerate(fwi_lats):
        for j, lon in enumerate(fwi_lons):
            ii = np.argmin(np.abs(bm_lats - lat))
            jj = np.argmin(np.abs(bm_lons - lon))
            mask_regrid[i, j] = burn_mask[ii, jj]
    return mask_regrid


def apply_tf(ds_fwi):
    """Apply transfer function; non-burnable cells get burn probability = 0."""
    vals = ds_fwi["fwi"].values

    # Get burn mask on the correct grid for this FWI dataset
    fwi_lats = ds_fwi.lat.values
    fwi_lons = ds_fwi.lon.values
    bm = get_burn_mask_for_grid(fwi_lats, fwi_lons)

    bp = np.zeros_like(vals, dtype=np.float32)
    burnable_3d = np.broadcast_to(bm[:, :, np.newaxis], vals.shape)
    valid = burnable_3d & ~np.isnan(vals)
    bp[valid] = lr_model.predict_proba(vals[valid].reshape(-1, 1))[:, 1] * cf

    # NaN where FWI is NaN
    bp[np.isnan(vals)] = np.nan
    return xr.DataArray(bp, coords=ds_fwi["fwi"].coords, dims=ds_fwi["fwi"].dims)


def annual_burn_prob(bp_da, years):
    annual = []
    for year in years:
        bp_yr = bp_da.sel(time=bp_da.time.dt.year == year)
        if bp_yr.sizes["time"] == 0: continue
        annual.append(1 - (1 - bp_yr).prod(dim="time"))
    ds = xr.concat(annual, dim="year")
    ds["year"] = list(years)
    return ds


# ── GEV fit and return periods ─────────────────────────────────────────────────
def fit_gev_return_levels(annual_bp_spatial_mean, rp_list=(2,5,10,20,50,100,200,500)):
    params = stats.genextreme.fit(annual_bp_spatial_mean)
    levels = {}
    for rp in rp_list:
        levels[rp] = float(stats.genextreme.ppf(1 - 1/rp, *params))
    return params, levels


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rp_list = [2, 5, 10, 20, 50, 100, 200, 500]

    # ── ERA5-Land historical ───────────────────────────────────────────────────
    ds_era5_abp = xr.open_dataset(
        os.path.join(RES_DIR, "ERA5-Land_historical_annual_burn_probability.nc"))
    era5_mean = np.nanmean(ds_era5_abp["annual_burn_probability"].values, axis=(1,2))
    gev_era5, rlevels_era5 = fit_gev_return_levels(era5_mean, rp_list)
    print("ERA5-Land return levels:")
    for rp, lv in rlevels_era5.items():
        print(f"  1-in-{rp:4d}: {lv:.5f} ({lv*100:.3f}%)")

    # ── CMIP6 historical + future ──────────────────────────────────────────────
    all_results = {"ERA5-Land_historical": rlevels_era5}
    annual_bp_store = {}

    for model in MODELS:
        for scenario in ["historical"] + list(SCENARIOS):
            tag      = f"{model}_{scenario}"
            fwi_path = os.path.join(FWI_DIR, f"{tag}_fwi.nc")
            out_path = os.path.join(RES_DIR, f"{tag}_annual_burn_probability.nc")

            if not os.path.exists(fwi_path):
                print(f"WARNING: {fwi_path} not found, skipping")
                continue

            if os.path.exists(out_path):
                print(f"{tag}: loading existing annual burn probability")
                abp = xr.open_dataset(out_path)["annual_burn_probability"]
            else:
                print(f"{tag}: computing annual burn probability...")
                ds_fwi = xr.open_dataset(fwi_path)
                # Normalise time
                ds_fwi["time"] = ds_fwi.time.values.astype("datetime64[D]").astype("datetime64[ns]")
                bp_da = apply_tf(ds_fwi)
                years = HIST_YEARS if scenario == "historical" else FUTURE_YEARS
                abp   = annual_burn_prob(bp_da, years)
                if os.path.exists(out_path): os.remove(out_path)
                abp.to_dataset(name="annual_burn_probability").to_netcdf(out_path)
                print(f"  Saved: {out_path}")

            spatial_mean = np.nanmean(abp.values, axis=(1,2))
            _, rlevels   = fit_gev_return_levels(spatial_mean, rp_list)
            all_results[tag] = rlevels
            annual_bp_store[tag] = spatial_mean

            print(f"{tag}: mean={spatial_mean.mean():.4f}, "
                  f"1-in-10={rlevels[10]:.5f}, 1-in-100={rlevels[100]:.5f}")

    # ── Return period table ────────────────────────────────────────────────────
    rows = []
    for tag, rlevels in all_results.items():
        row = {"dataset": tag}
        row.update({f"rp_{rp}yr": rlevels[rp] for rp in rp_list})
        rows.append(row)
    rp_df = pd.DataFrame(rows)
    rp_df.to_csv(os.path.join(RES_DIR, "return_period_table.csv"), index=False)
    print("\nReturn period table saved")

    # ── Ensemble return period table ───────────────────────────────────────────
    ensemble_rows = []
    for scenario in ["historical"] + list(SCENARIOS):
        for rp in rp_list:
            key  = f"rp_{rp}yr"
            vals = [all_results[f"{m}_{scenario}"][rp]
                    for m in MODELS if f"{m}_{scenario}" in all_results]
            if not vals: continue
            ensemble_rows.append({
                "scenario": scenario, "return_period": rp,
                "ensemble_mean":  np.mean(vals),
                "ensemble_p5":    np.percentile(vals, 5),
                "ensemble_p95":   np.percentile(vals, 95),
                "era5_land":      rlevels_era5.get(rp, np.nan),
            })
    ens_df = pd.DataFrame(ensemble_rows)
    ens_df.to_csv(os.path.join(RES_DIR, "ensemble_return_period_table.csv"), index=False)
    print("Ensemble return period table saved")

    # ── Return period plot ─────────────────────────────────────────────────────
    rp_range       = np.logspace(0, 2.7, 100)
    exceedance     = 1 / rp_range
    era5_curve     = stats.genextreme.ppf(1-exceedance, *gev_era5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"CanESM5":"#e41a1c","MPI-ESM1-2-HR":"#377eb8","ACCESS-CM2":"#4daf4a"}

    # Historical comparison
    ax = axes[0]
    ax.plot(rp_range, era5_curve, color="black", linewidth=2.5, label="ERA5-Land")
    for model in MODELS:
        tag = f"{model}_historical"
        if tag not in all_results: continue
        _, _, _, _, _, tag_curve = [None]*5 + [None]
        sm     = annual_bp_store.get(tag)
        if sm is None: continue
        params = stats.genextreme.fit(sm)
        curve  = stats.genextreme.ppf(1-exceedance, *params)
        ax.plot(rp_range, curve, color=colors[model], linewidth=1.5,
                linestyle="--", label=f"{model} hist")
    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)"); ax.set_ylabel("Annual Burn Probability")
    ax.set_title("Return Periods: Historical"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Climate change signal — ensemble mean historical vs future
    ax = axes[1]
    ax.plot(rp_range, era5_curve, color="black", linewidth=2.5, label="ERA5-Land (1980-2014)")
    for scenario, color, label in [("historical","#377eb8","Historical (1980-2014)"),
                                    ("ssp245","#ff7f00","SSP2-4.5 (2015-2100)"),
                                    ("ssp585","#e41a1c","SSP5-8.5 (2015-2100)")]:
        vals = [annual_bp_store.get(f"{m}_{scenario}") for m in MODELS
                if f"{m}_{scenario}" in annual_bp_store]
        vals = [v for v in vals if v is not None]
        if not vals: continue
        curves = []
        for v in vals:
            p = stats.genextreme.fit(v)
            curves.append(stats.genextreme.ppf(1-exceedance, *p))
        mean_curve = np.mean(curves, axis=0)
        p5_curve   = np.percentile(curves, 5, axis=0)
        p95_curve  = np.percentile(curves, 95, axis=0)
        ax.plot(rp_range, mean_curve, color=color, linewidth=2, label=label)
        ax.fill_between(rp_range, p5_curve, p95_curve, color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)"); ax.set_ylabel("Annual Burn Probability")
    ax.set_title("Climate Change Signal (Ensemble)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Wildfire Return Period Analysis: Central BC", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "return_period_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Return period plots saved")

    # ── Climate change signal summary ──────────────────────────────────────────
    print("\n=== Climate Change Signal (Ensemble Mean) ===")
    print(f"{'Return Period':>15} | {'ERA5-Land':>10} | {'Hist':>10} | {'SSP2-4.5':>10} | {'SSP5-8.5':>10}")
    print("-" * 65)
    for rp in [2, 10, 50, 100]:
        row = [f"1-in-{rp}"]
        row.append(f"{rlevels_era5[rp]:.5f}")
        for scenario in ["historical","ssp245","ssp585"]:
            vals = [all_results.get(f"{m}_{scenario}",{}).get(rp, np.nan) for m in MODELS]
            vals = [v for v in vals if not np.isnan(v)]
            row.append(f"{np.mean(vals):.5f}" if vals else "N/A")
        print(f"{row[0]:>15} | {' | '.join(f'{r:>10}' for r in row[1:])}")

    print("\nReturn period analysis complete")

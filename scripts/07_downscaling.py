"""
07_downscaling.py
Spatially downscales coarse GCM-derived burn probability to 30m resolution
using Mulverhill et al. (2025) burn probability as spatial weights.

Methodology:
- Our pipeline (scripts 01-06) produces annual burn probability at 0.25 deg
  CMIP6 resolution (~25km). This captures the temporal signal — how fire risk
  changes year to year and across climate scenarios — but is too coarse for
  asset-level financial risk applications.
- Mulverhill et al. (2025) provides 30m burn probability for Canada's forested
  ecosystems under historical and future scenarios, based on climate, vegetation,
  and topography. Values represent relative hazard (0-100 scale), not annual
  probability. Non-treed pixels = 255.
- Disaggregation approach: for each coarse cell, the Mulverhill spatial pattern
  is normalised to sum to 1, then used to distribute the coarse annual burn
  probability across 30m sub-pixels. Each 30m pixel receives a fraction of the
  coarse probability proportional to its relative hazard within the cell.
- Historical period uses Mulverhill baseline (1991-2020) as spatial weights.
- Future period (SSP5-8.5 2081-2100) uses Mulverhill SSP5-8.5 2081-2100 weights,
  capturing climate-driven shifts in spatial fire pattern under strong warming.

Validation:
- Compares downscaled historical burn probability against Mulverhill baseline
  at 30m to assess spatial agreement.
- Computes Pearson correlation and RMSE over forested pixels in the domain.

Outputs:
  data/outputs/results/
    ERA5-Land_historical_burn_probability_30m.tif              (unsmoothed)
    ERA5-Land_historical_burn_probability_30m_smoothed.tif     (presentation)
    ACCESS-CM2_ssp585_2081-2100_burn_probability_30m.tif       (unsmoothed)
    ACCESS-CM2_ssp585_2081-2100_burn_probability_30m_smoothed.tif (presentation)
    validation_correlation.csv
  data/outputs/figures/
    downscaling_validation.png
    downscaling_climate_signal_30m.png

References:
  Mulverhill et al. (2025). Canadian Journal of Remote Sensing 51(1).
  Mulverhill et al. (2024). ISPRS J. Photogrammetry and Remote Sensing 209, 279-295.
"""

import xarray as xr
import numpy as np
import rioxarray as rxr
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pickle
import yaml
import os

with open("config.yml") as f:
    config = yaml.safe_load(f)

loc     = config["location"]
RES_DIR = config["data"]["outputs"]["results"]
FIG_DIR = config["data"]["outputs"]["figures"]
BP_BASE = config["data"]["raw"]["burn_probability"]["baseline"]
BP_SSP5 = config["data"]["raw"]["burn_probability"]["ssp585_2081_2100"]
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Domain bounds in EPSG:3978 (Mulverhill native CRS)
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)
X_MIN, Y_MIN = _transformer.transform(loc["lon_min"], loc["lat_min"])
X_MAX, Y_MAX = _transformer.transform(loc["lon_max"], loc["lat_max"])


# ── Load and clip Mulverhill raster ───────────────────────────────────────────
def load_mulverhill(tif_path):
    """
    Load Mulverhill burn probability raster clipped to domain.
    Returns float32 array with NaN for non-treed pixels (255).
    """
    ds = rxr.open_rasterio(tif_path, chunks=512)
    ds_clip = ds.rio.clip_box(minx=X_MIN, miny=Y_MIN, maxx=X_MAX, maxy=Y_MAX)
    arr = ds_clip.squeeze().values.astype(np.float32)
    arr[arr == 255] = np.nan  # non-treed -> NaN
    print(f"  Mulverhill loaded: shape={arr.shape}, "
          f"valid pixels={int(~np.isnan(arr)).sum() if arr.ndim==1 else int((~np.isnan(arr)).sum())}, "
          f"mean={np.nanmean(arr):.2f}")
    return arr, ds_clip


# ── Build spatial weight tiles per coarse cell ────────────────────────────────
def build_weight_tiles(mulv_arr, mulv_ds, coarse_lats, coarse_lons):
    """
    For each coarse grid cell, extract Mulverhill pixels and normalise to
    sum-to-1 spatial weights.

    Returns:
        weight_tiles: dict keyed by (i,j) -> (yi, xi, weights_1d)
            yi, xi: row/col indices into mulv_arr for pixels in this cell
            weights_1d: normalised weights summing to 1
        mulv_shape: (nrows, ncols) of the clipped Mulverhill array
    """
    print("Building spatial weight tiles...")
    inv_transformer = Transformer.from_crs("EPSG:3978", "EPSG:4326", always_xy=True)

    xs = mulv_ds.x.values
    ys = mulv_ds.y.values

    # Coarse cell half-widths in degrees
    dlat = abs(coarse_lats[1] - coarse_lats[0])
    dlon = abs(coarse_lons[1] - coarse_lons[0])

    # Reproject coarse cell bounds to EPSG:3978 for indexing
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)

    weight_tiles = {}
    n_empty = 0

    for i, lat in enumerate(coarse_lats):
        for j, lon in enumerate(coarse_lons):
            # Cell bounds in EPSG:4326
            lat0 = lat - dlat / 2
            lat1 = lat + dlat / 2
            lon0 = lon - dlon / 2
            lon1 = lon + dlon / 2

            # Reproject corners to EPSG:3978
            cx0, cy0 = fwd.transform(lon0, lat0)
            cx1, cy1 = fwd.transform(lon1, lat1)

            xi = np.where((xs >= min(cx0, cx1)) & (xs <= max(cx0, cx1)))[0]
            yi = np.where((ys >= min(cy0, cy1)) & (ys <= max(cy0, cy1)))[0]

            if len(xi) == 0 or len(yi) == 0:
                weight_tiles[(i, j)] = None
                n_empty += 1
                continue

            patch = mulv_arr[np.ix_(yi, xi)]  # (ny, nx)
            valid_mask = ~np.isnan(patch)

            if valid_mask.sum() == 0:
                # No treed pixels in this cell
                weight_tiles[(i, j)] = None
                n_empty += 1
                continue

            # Normalise weights so mean=1 (preserves coarse probability as cell mean)
            # BP_pixel = BP_coarse * (mulv_pixel / mulv_mean)
            # This way: mean(BP_pixel) = BP_coarse, not BP_coarse/n_pixels
            vals = patch[valid_mask]
            vals_mean = vals.mean()
            if vals_mean < 1e-9:
                # All pixels have ~zero hazard — assign uniform weight
                weights = np.ones(len(vals)) / len(vals)
            else:
                weights = vals / vals_mean  # relative weights, mean=1

            # Store flat indices into the full mulv_arr for fast assignment
            yy, xx = np.where(valid_mask)
            yi_abs = yi[yy]
            xi_abs = xi[xx]

            weight_tiles[(i, j)] = (yi_abs, xi_abs, weights)

    n_total = len(coarse_lats) * len(coarse_lons)
    print(f"  Weight tiles: {n_total - n_empty}/{n_total} cells have treed pixels")
    return weight_tiles, mulv_arr.shape


# ── Downscale one annual burn probability field ───────────────────────────────
def downscale_annual(annual_bp_arr, weight_tiles, out_shape, coarse_lats, coarse_lons):
    """
    Distribute coarse annual burn probability to 30m pixels using weight tiles.

    annual_bp_arr: (n_lat, n_lon) coarse annual burn probability
    Returns: (nrows, ncols) 30m burn probability array
    """
    out = np.full(out_shape, np.nan, dtype=np.float32)

    for i, lat in enumerate(coarse_lats):
        for j, lon in enumerate(coarse_lons):
            tile = weight_tiles.get((i, j))
            if tile is None:
                continue
            coarse_val = annual_bp_arr[i, j]
            if np.isnan(coarse_val):
                continue
            yi_abs, xi_abs, weights = tile
            out[yi_abs, xi_abs] = weights * coarse_val

    return out


# ── Downscale full time series → mean annual burn probability ─────────────────
def downscale_mean_annual(ds_abp, weight_tiles, out_shape, coarse_lats, coarse_lons):
    """
    Downscale each year's annual burn probability and average over all years.
    Returns 30m mean annual burn probability array.
    """
    # Annual burn probability dim is named 'year' (from compute_annual_burn_prob)
    var_name = "annual_burn_probability"
    arr = ds_abp[var_name].values  # (years, lat, lon)
    n_years = arr.shape[0]
    print(f"  Downscaling {n_years} years, coarse shape {arr.shape}...")

    accumulator = np.zeros(out_shape, dtype=np.float64)
    count       = np.zeros(out_shape, dtype=np.int32)

    for t in range(n_years):
        frame = downscale_annual(arr[t], weight_tiles, out_shape, coarse_lats, coarse_lons)
        valid = ~np.isnan(frame)
        accumulator[valid] += frame[valid]
        count[valid] += 1

    mean_bp = np.where(count > 0, accumulator / count, np.nan).astype(np.float32)
    # Clip to valid probability range — weights can push extreme pixels above 1
    mean_bp = np.clip(mean_bp, 0.0, 1.0)
    print(f"  Downscaled mean annual BP: mean={np.nanmean(mean_bp):.6f} "
          f"({np.nanmean(mean_bp)*100:.4f}%), "
          f"max={np.nanmax(mean_bp):.6f}")
    return mean_bp


# ── Save 30m GeoTIFF ──────────────────────────────────────────────────────────
def save_geotiff(arr, mulv_ds, out_path):
    """Save 30m array as GeoTIFF in EPSG:3978."""
    da = xr.DataArray(
        arr[np.newaxis],
        dims=["band", "y", "x"],
        coords={"band": [1], "y": mulv_ds.y.values, "x": mulv_ds.x.values}
    )
    da = da.rio.write_crs("EPSG:3978")
    da = da.rio.write_nodata(np.nan)
    if os.path.exists(out_path):
        os.remove(out_path)
    da.rio.to_raster(out_path, dtype="float32")
    print(f"  Saved: {out_path}")



# ── Smooth tile boundaries ────────────────────────────────────────────────────
def smooth_tile_boundaries(arr, coarse_lats, coarse_lons, mulv_ds,
                           sigma_pixels=600):
    """
    Remove sharp tile boundaries from downscaled output using a constrained
    Gaussian smoothing approach.

    The tiling artifact arises because adjacent CMIP6 cells have different
    burn probabilities, producing sharp discontinuities at cell edges. Non-treed
    gaps (NaN pixels) fragment the domain into islands that resist short-kernel
    smoothing. The kernel must be wide enough (~1 full CMIP6 cell width) to
    genuinely bridge across these gaps.

    Approach:
      1. Fill NaN gaps with nearest-neighbour interpolation so the kernel
         propagates across non-treed areas without edge halos.
      2. Apply Gaussian filter with sigma large enough to span tile boundaries
         (~0.25 deg CMIP6 cell = ~833 pixels at 30m; sigma=600 blends edges
         while retaining broad spatial gradients).
      3. Reapply original NaN mask — non-treed pixels return to NaN.

    sigma_pixels: Gaussian sigma in 30m pixels.
      600 = 18km kernel — spans most of a CMIP6 cell, eliminates tile pattern.
      300 = 9km — partial blending, some tile structure may remain.
      100 = 3km — within-cell smoothing only, tile boundaries still visible.

    The unsmoothed GeoTIFF is always saved alongside the smoothed version.
    Use the unsmoothed file for quantitative analysis; smoothed for presentation.
    """
    from scipy.ndimage import distance_transform_edt

    print(f"  Smoothing tile boundaries (sigma={sigma_pixels} pixels = "
          f"{sigma_pixels * 30 / 1000:.1f}km)...")

    nan_mask = np.isnan(arr)

    # Step 1: nearest-neighbour fill across NaN gaps
    if nan_mask.any():
        _, nearest_idx = distance_transform_edt(nan_mask, return_indices=True)
        arr_filled = arr[nearest_idx[0], nearest_idx[1]]
    else:
        arr_filled = arr.copy()

    # Step 2: Gaussian smoothing — no NaN-aware weighting needed since gaps
    # are filled; all pixels contribute equally to the kernel
    smoothed = gaussian_filter(arr_filled.astype(np.float64),
                               sigma=sigma_pixels).astype(np.float32)

    # Step 3: reapply original NaN mask
    smoothed = np.clip(smoothed, 0.0, 1.0)
    smoothed[nan_mask] = np.nan

    print(f"  Smoothed mean: {np.nanmean(smoothed)*100:.4f}% "
          f"(raw: {np.nanmean(arr)*100:.4f}%)")
    return smoothed
def validate(downscaled_hist, mulv_baseline_arr, fig_dir):
    """
    Compare downscaled historical burn probability against Mulverhill baseline.

    Both arrays are in the same 30m grid. Mulverhill is a relative hazard index
    (0-100), downscaled is absolute annual probability. We compare spatial
    patterns via Pearson correlation on forested pixels.
    """
    # Valid where both arrays have data
    valid = ~np.isnan(downscaled_hist) & ~np.isnan(mulv_baseline_arr)
    n_valid = valid.sum()
    print(f"  Validation pixels: {n_valid:,}")

    if n_valid < 100:
        print("  WARNING: insufficient valid pixels for validation")
        return

    ds_vals  = downscaled_hist[valid]
    mul_vals = mulv_baseline_arr[valid] / 100.0  # normalise to 0-1

    corr = float(np.corrcoef(ds_vals, mul_vals)[0, 1])
    rmse = float(np.sqrt(np.mean((ds_vals - mul_vals) ** 2)))
    print(f"  Pearson r = {corr:.3f}")
    print(f"  RMSE = {rmse:.4f} (note: different scales)")

    # Save validation stats
    pd.DataFrame([{"pearson_r": corr, "rmse": rmse, "n_pixels": n_valid}]).to_csv(
        os.path.join(RES_DIR, "validation_correlation.csv"), index=False)

    # Scatter plot (subsample for speed)
    idx = np.random.choice(n_valid, size=min(50000, n_valid), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(mul_vals[idx], ds_vals[idx], alpha=0.05, s=1, color="steelblue")
    axes[0].set_xlabel("Mulverhill Baseline (normalised 0-1)")
    axes[0].set_ylabel("Downscaled Annual Burn Probability")
    axes[0].set_title(f"Spatial Agreement: r={corr:.3f}")
    axes[0].grid(True, alpha=0.3)

    # Spatial maps side by side (subsample to manageable size)
    step = max(1, downscaled_hist.shape[0] // 500)
    axes[1].imshow(downscaled_hist[::step, ::step], cmap="YlOrRd",
                   vmin=0, vmax=np.nanpercentile(downscaled_hist, 99),
                   aspect="auto")
    axes[1].set_title("Downscaled Historical Burn Probability (30m)")
    axes[1].axis("off")

    plt.suptitle("Downscaling Validation: ERA5-Land vs Mulverhill Baseline", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "downscaling_validation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Validation plot saved")
    return corr, rmse


# ── Climate signal plot ────────────────────────────────────────────────────────
def plot_climate_signal(hist_30m, fut_30m, fig_dir):
    """Plot historical vs future 30m burn probability and their ratio."""
    step = max(1, hist_30m.shape[0] // 500)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmax_bp = max(np.nanpercentile(hist_30m, 99),
                  np.nanpercentile(fut_30m, 99))

    im0 = axes[0].imshow(hist_30m[::step, ::step], cmap="YlOrRd",
                         vmin=0, vmax=vmax_bp, aspect="auto")
    plt.colorbar(im0, ax=axes[0], label="Annual Burn Probability")
    axes[0].set_title("Historical (ERA5-Land, 1980-2014)")
    axes[0].axis("off")

    im1 = axes[1].imshow(fut_30m[::step, ::step], cmap="YlOrRd",
                         vmin=0, vmax=vmax_bp, aspect="auto")
    plt.colorbar(im1, ax=axes[1], label="Annual Burn Probability")
    axes[1].set_title("Future (ACCESS-CM2 SSP5-8.5, 2081-2100)")
    axes[1].axis("off")

    # Ratio — future / historical, clipped for display
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(hist_30m > 1e-6, fut_30m / hist_30m, np.nan)
    ratio = np.clip(ratio, 0, 5)

    cmap_div = plt.cm.RdBu_r
    im2 = axes[2].imshow(ratio[::step, ::step], cmap=cmap_div,
                         vmin=0, vmax=5, aspect="auto")
    plt.colorbar(im2, ax=axes[2], label="Future / Historical ratio")
    axes[2].set_title("Climate Change Signal (ratio, capped at 5×)")
    axes[2].axis("off")

    plt.suptitle("30m Burn Probability: Historical vs SSP5-8.5 2081-2100\n"
                 "Central BC | Downscaled using Mulverhill et al. (2025)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "downscaling_climate_signal_30m.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Climate signal plot saved")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load Mulverhill rasters ────────────────────────────────────────────────
    print("Loading Mulverhill baseline...")
    mulv_base_arr, mulv_base_ds = load_mulverhill(BP_BASE)

    print("Loading Mulverhill SSP5-8.5 2081-2100...")
    mulv_ssp5_arr, mulv_ssp5_ds = load_mulverhill(BP_SSP5)

    # ── Load coarse annual burn probability ────────────────────────────────────
    print("\nLoading coarse annual burn probability...")
    ds_hist = xr.open_dataset(
        os.path.join(RES_DIR, "ERA5-Land_historical_annual_burn_probability.nc"))
    ds_fut = xr.open_dataset(
        os.path.join(RES_DIR, "ACCESS-CM2_ssp585_annual_burn_probability.nc"))

    # ERA5-Land and CMIP6 are on different grids — load lats/lons separately
    hist_lats = ds_hist.lat.values
    hist_lons = ds_hist.lon.values
    fut_lats  = ds_fut.lat.values
    fut_lons  = ds_fut.lon.values
    print(f"  ERA5-Land grid: {len(hist_lats)} lat x {len(hist_lons)} lon")
    print(f"  CMIP6 grid:     {len(fut_lats)} lat x {len(fut_lons)} lon")

    # ── Build weight tiles ─────────────────────────────────────────────────────
    print("\nBuilding historical weight tiles (Mulverhill baseline, ERA5-Land grid)...")
    wt_hist, out_shape = build_weight_tiles(
        mulv_base_arr, mulv_base_ds, hist_lats, hist_lons)

    print("\nBuilding future weight tiles (Mulverhill SSP5-8.5 2081-2100, CMIP6 grid)...")
    wt_fut, _ = build_weight_tiles(
        mulv_ssp5_arr, mulv_ssp5_ds, fut_lats, fut_lons)

    # ── Downscale historical ───────────────────────────────────────────────────
    hist_out          = os.path.join(RES_DIR, "ERA5-Land_historical_burn_probability_30m.tif")
    hist_out_smoothed = os.path.join(RES_DIR, "ERA5-Land_historical_burn_probability_30m_smoothed.tif")
    if not os.path.exists(hist_out):
        print("\nDownscaling ERA5-Land historical...")
        hist_30m = downscale_mean_annual(
            ds_hist, wt_hist, out_shape, hist_lats, hist_lons)
        save_geotiff(hist_30m, mulv_base_ds, hist_out)
        hist_30m_smoothed = smooth_tile_boundaries(hist_30m, hist_lats, hist_lons, mulv_base_ds)
        save_geotiff(hist_30m_smoothed, mulv_base_ds, hist_out_smoothed)
    else:
        print("\nHistorical 30m exists, loading...")
        hist_30m = rxr.open_rasterio(hist_out).squeeze().values
        if not os.path.exists(hist_out_smoothed):
            hist_30m_smoothed = smooth_tile_boundaries(hist_30m, hist_lats, hist_lons, mulv_base_ds)
            save_geotiff(hist_30m_smoothed, mulv_base_ds, hist_out_smoothed)
        else:
            hist_30m_smoothed = rxr.open_rasterio(hist_out_smoothed).squeeze().values

    # ── Downscale future ───────────────────────────────────────────────────────
    # Use only years 2081-2100 to match the Mulverhill epoch
    fut_out          = os.path.join(RES_DIR, "ACCESS-CM2_ssp585_2081-2100_burn_probability_30m.tif")
    fut_out_smoothed = os.path.join(RES_DIR, "ACCESS-CM2_ssp585_2081-2100_burn_probability_30m_smoothed.tif")
    if not os.path.exists(fut_out):
        print("\nDownscaling ACCESS-CM2 SSP5-8.5 2081-2100...")
        ds_fut_epoch = ds_fut.sel(year=slice(2081, 2100))
        fut_30m = downscale_mean_annual(
            ds_fut_epoch, wt_fut, out_shape, fut_lats, fut_lons)
        save_geotiff(fut_30m, mulv_ssp5_ds, fut_out)
        fut_30m_smoothed = smooth_tile_boundaries(fut_30m, fut_lats, fut_lons, mulv_ssp5_ds)
        save_geotiff(fut_30m_smoothed, mulv_ssp5_ds, fut_out_smoothed)
    else:
        print("\nFuture 30m exists, loading...")
        fut_30m = rxr.open_rasterio(fut_out).squeeze().values
        if not os.path.exists(fut_out_smoothed):
            fut_30m_smoothed = smooth_tile_boundaries(fut_30m, fut_lats, fut_lons, mulv_ssp5_ds)
            save_geotiff(fut_30m_smoothed, mulv_ssp5_ds, fut_out_smoothed)
        else:
            fut_30m_smoothed = rxr.open_rasterio(fut_out_smoothed).squeeze().values

    # ── Validation ────────────────────────────────────────────────────────────
    # Validate against unsmoothed — smoothing is cosmetic, not analytical
    print("\nValidating historical downscaling against Mulverhill baseline...")
    validate(hist_30m, mulv_base_arr, FIG_DIR)

    # ── Climate signal plot — use smoothed for presentation ───────────────────
    print("\nPlotting climate change signal (smoothed)...")
    plot_climate_signal(hist_30m_smoothed, fut_30m_smoothed, FIG_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Downscaling Summary ===")
    print(f"Output resolution: 30m (EPSG:3978)")
    hist_mean = float(np.nanmean(hist_30m))
    fut_mean  = float(np.nanmean(fut_30m))
    print(f"Historical mean annual BP: {hist_mean:.6f} ({hist_mean*100:.4f}%)")
    print(f"Future mean annual BP:     {fut_mean:.6f} ({fut_mean*100:.4f}%)")
    if hist_mean > 1e-9:
        print(f"Future/Historical ratio:   {fut_mean/hist_mean:.2f}x")
    print(f"\nOutputs saved:")
    print(f"  Unsmoothed (quantitative use): *_30m.tif")
    print(f"  Smoothed (presentation use):   *_30m_smoothed.tif")
    print("\nDownscaling complete")

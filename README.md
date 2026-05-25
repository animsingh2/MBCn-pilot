# Wildfire Physical Risk Pipeline
## Central BC Wildfire Hazard Quantification

A methodology for quantifying wildfire physical risk from climate data,
following the hazard→exposure→vulnerability chain used by climate risk
practitioners at financial institutions.

---

## Overview

This pipeline estimates **annual burn probability**, **return period curves**,
and **30m spatially downscaled burn probability maps** for a study region in
central British Columbia under historical and future climate scenarios. The
methodology uses entirely open-source data and tools and is designed to produce
outputs comparable to those of commercial climate risk vendors (e.g. Jupiter
Intelligence) within the constraints of open data.

**Study region:** Central BC (51-54°N, 120-125°W)
**Historical period:** 1980-2014
**Future periods:** 2015-2100 (SSP2-4.5 and SSP5-8.5)
**Climate models:** CanESM5, MPI-ESM1-2-HR, ACCESS-CM2
**Output resolution:** 0.1° ERA5-Land, 0.25° CMIP6, 30m downscaled

---

## Pipeline

```
01_download_era5_land.py     Download hourly ERA5-Land, extract solar noon values
02_download_cmip6.py         Download NEX-GDDP-CMIP6 for all models/scenarios
03_bias_correction.py        Bias-correct CMIP6 against ERA5-Land (LOCI + MBCn)
04_fwi_computation.py        Compute FWI for ERA5-Land and all CMIP6 runs
05_transfer_function.py      Fit FWI → burn probability transfer function
06_return_periods.py         Compute return period curves and climate change signal
07_downscaling.py            Downscale burn probability to 30m using Mulverhill et al. (2025)
```

---

## Methodology

### Climate Data
- **ERA5-Land** (ECMWF, 0.1°/~9km): observational reference for bias correction.
  Downloaded at hourly resolution; solar noon values extracted per grid cell
  using astronomical solar noon (Duffie et al. 2020). Temperature and wind
  interpolated between hourly values bracketing solar noon. Precipitation taken
  as 23:00 UTC accumulation (ERA5-Land resets at midnight UTC).
- **NEX-GDDP-CMIP6** (0.25°/~25km): daily climate model projections from NASA
  AWS S3. Three models selected to span the CMIP6 climate sensitivity range:
  CanESM5 (high sensitivity), MPI-ESM1-2-HR (medium), ACCESS-CM2 (medium-high).

### Bias Correction
Applied to CMIP6 historical (1980-2014) against ERA5-Land reference:
1. **LOCI** (Local Intensity Scaling): corrects precipitation wet day frequency
   (Themessl et al. 2012)
2. **MBCn** (Multivariate Bias Correction): corrects all four variables
   simultaneously (tasmax, hurs, sfcWind, pr), preserving co-occurrence
   structure between temperature, humidity, and wind (Cannon 2018). Validated
   specifically on Canadian FWI applications.

For future scenarios (2015-2100), MBCn is applied in sequential 35-year windows
(2015-2049, 2050-2084, 2085-2100) matching the historical training period length.
This avoids the MBCn rank-reordering collapse that occurs when the simulation
period exceeds the reference period length (Cannon 2018).

### FWI Computation
Canadian Forest Fire Weather Index system (Van Wagner 1987) implemented via
xclim. Computed year-by-year to reinitialise carry-over moisture codes (DC, DMC,
FFMC) each fire season. Inputs at local solar noon per grid cell. Wind speed
converted from m/s to km/h before computation.

### Transfer Function
Maps daily FWI to daily burn probability using a hybrid NBAC+NFDB approach:
- **NBAC** (National Burned Area Composite, 1980-2014): satellite-derived fire
  perimeters at 30m. Identifies which FWI grid cells burned each year.
- **NFDB** (National Fire Database): fire ignition point records with dates.
  Only ignitions spatially confirmed by NBAC perimeters are used, assigning
  fire occurrence to specific days.
- **VLCE2** land cover (Hermosilla et al. 2022): non-burnable cells (water,
  snow/ice, rock, wetland — VLCE2 classes 20, 31, 32, 80) excluded from
  training using modal class per ERA5-Land grid cell.
- **Logistic regression** with Bayesian prevalence correction (correction
  factor applied post-hoc to align model output with true fire prevalence).
- NBAC centroids computed in native projected CRS (Canada Lambert) before
  reprojection to EPSG:4326 to avoid geographic centroid distortion for
  large BC fire polygons.

Transfer function performance: AUC = 0.774. The primary limitation is the
FWI-only predictor — fuel type stratification (e.g. CFS National Fuel Type
layer) would improve AUC to an estimated 0.82-0.85.

### Return Periods
GEV (Generalised Extreme Value) distribution fitted to annual burn probability
time series (spatial mean across domain). Multi-model ensemble with 5th/95th
percentile uncertainty bounds across 3 CMIP6 models. Return periods reported
up to 1-in-50 for historical (35-year record) and 1-in-100 for future
(86-year record). Extrapolation beyond these thresholds is not statistically
supported by the available record length.

### Spatial Downscaling
Coarse CMIP6 burn probability (0.25°) disaggregated to 30m using Mulverhill
et al. (2025) as spatial weights:
- **Mulverhill et al. (2025)**: 30m projected burn probability for Canada's
  forested ecosystems under SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5 for four
  future epochs (2021-2040, 2041-2060, 2061-2080, 2081-2100) plus baseline
  1991-2020. Values represent relative fire hazard (0-100 scale) based on
  climate, vegetation, and topography. Non-treed pixels = 255.
- **Disaggregation method**: for each coarse cell, Mulverhill pixel values are
  normalised by their cell mean (weights with mean=1). Each 30m pixel receives
  BP_coarse × (mulv_pixel / mulv_mean), preserving the coarse mean probability
  while distributing spatial variability from the high-resolution hazard layer.
- Historical period uses Mulverhill baseline (1991-2020) as spatial weights.
  Future period uses epoch-matched Mulverhill weights (SSP5-8.5 2081-2100),
  capturing climate-driven shifts in spatial fire pattern under strong warming.
- Validation against Mulverhill baseline: Pearson r = 0.807 over 51M forested
  pixels in the domain.

---

## Data Requirements

### Downloaded automatically
- ERA5-Land hourly (Copernicus CDS — requires free registration at
  https://cds.climate.copernicus.eu)
- NEX-GDDP-CMIP6 (NASA AWS S3 — no registration required)

### Downloaded manually
Place in the following directories before running:

```
data/raw/nbac/
    NBAC fire perimeters (shapefile)
    https://cwfis.cfs.nrcan.gc.ca/datamart/download/nbac

data/raw/nfdb/
    NFDB fire point records (shapefile)
    https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt

data/raw/landcover/
    VLCE2 annual land cover GeoTIFF (any year 1984-2022)
    https://opendata.nfis.org — "Land cover 1984-2022 Version 2"

data/raw/wildfire_landsat/
    CA Forest Wildfire 1985-2022 change year GeoTIFF
    https://opendata.nfis.org — "CA Forest Wildfire 1985-2022"

data/raw/burn_probability/baseline/
    CA_Forest_Burn_Probability_baseline_1991-2020.tif (18.4 GB)
    https://opendata.nfis.org — "Projected Burn Probability 2020-2100"

data/raw/burn_probability/ssp585_2081-2100/
    CA_Forest_Burn_Probability_SSP5-8.5_2081-2100.tif (18.9 GB)
    https://opendata.nfis.org — "Projected Burn Probability 2020-2100"
```

---

## Setup

```bash
pip install xarray xclim xsdba xesmf s3fs cdsapi ephem numpy pandas \
            matplotlib geopandas scikit-learn scipy pyyaml netCDF4 \
            h5netcdf rioxarray shapely rasterio pyproj
conda install -c conda-forge esmpy
```

Configure Copernicus CDS API key in `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: your-api-key-here
```

---

## Running

Run scripts in order from the project root directory:
```bash
python scripts/01_download_era5_land.py
python scripts/02_download_cmip6.py
python scripts/03_bias_correction.py
python scripts/04_fwi_computation.py
python scripts/05_transfer_function.py
python scripts/06_return_periods.py
python scripts/07_downscaling.py
```

Each script has skip logic — if outputs already exist they are skipped.
Scripts can be safely interrupted and restarted.

Script 07 currently demonstrates downscaling for ERA5-Land historical and
ACCESS-CM2 SSP5-8.5 2081-2100. To run additional scenarios or epochs, download
the corresponding Mulverhill GeoTIFFs and add their paths to config.yml.

---

## Outputs

```
data/processed/
  era5_land/                  Solar noon ERA5-Land daily files (1980-2014)
  cmip6_bias_corrected/       Bias-corrected CMIP6 historical (3 models)
  fwi/                        FWI + components for ERA5-Land and all CMIP6 runs

data/outputs/
  figures/                    Validation and results plots
  models/                     Transfer function model (includes burn mask)
  results/
    return_period_table.csv               Per-dataset GEV return levels
    ensemble_return_period_table.csv      Ensemble mean + 5th/95th percentile
    ERA5-Land_historical_burn_probability_30m.tif     30m historical
    ACCESS-CM2_ssp585_2081-2100_burn_probability_30m.tif  30m future
    validation_correlation.csv            Downscaling validation statistics
```

Primary outputs for financial risk applications:
- `ensemble_return_period_table.csv` — regional return period curves
- `*_burn_probability_30m.tif` — asset-level 30m burn probability maps

---

## Key Limitations

1. **Transfer function AUC 0.774**: FWI-only predictor. Adding CFS National
   Fuel Type layer as a predictor would improve AUC to ~0.82-0.85. Absolute
   burn probability values should be treated as indicative; the relative
   climate change signal is more robust than absolute magnitudes.

2. **Return period extrapolation**: GEV fitted to 35-year historical and
   86-year future records. Return periods beyond 1-in-50 (historical) and
   1-in-100 (future) are extrapolations with wide uncertainty. Vendor-grade
   return period estimation requires stochastic event sets (~10,000 synthetic
   years) which are not reproducible with open data alone.

3. **No fire spread modelling**: burn probability reflects ignition likelihood
   given FWI conditions, not actual burned area. Commercial vendors run spatial
   fire growth simulation (e.g. Prometheus, FARSITE) which accounts for
   topography, fuel continuity, and suppression. This is the primary
   methodological gap versus Jupiter Intelligence.

4. **Wind speed**: ERA5-Land underestimates surface winds over complex BC
   terrain due to coarse orographic representation. No open-data alternative
   exists for all four FWI input variables simultaneously.

5. **Downscaling scope**: script 07 demonstrates the method for one scenario
   (SSP5-8.5) and one future epoch (2081-2100). Full coverage requires
   downloading all Mulverhill epochs (~165 GB).

6. **Static vegetation**: Mulverhill spatial weights use contemporary (2020)
   forest conditions for all future periods. Climate-driven vegetation shifts
   (species migration, post-fire succession) are not captured.

---

## References

- Cannon A.J. (2018). Multivariate quantile mapping bias correction: an
  N-dimensional probability density function transform for climate model
  simulations of multiple variables. Climate Dynamics 50(1-2), 31-49.
- Hermosilla T. et al. (2022). Land cover classification in an era of big and
  open data. Remote Sensing of Environment 268, 112780.
- Mulverhill C. et al. (2024). Multidecadal mapping of status and trends in
  annual burn probability over Canada's forested ecosystems. ISPRS Journal of
  Photogrammetry and Remote Sensing 209, 279-295.
- Mulverhill C. et al. (2025). Projected Future Changes in Burn Probability in
  Canada's Forests and Communities Under Different Climate Change Scenarios.
  Canadian Journal of Remote Sensing 51(1).
- Themessl M.J. et al. (2012). Empirical-statistical downscaling and error
  correction of daily precipitation from regional climate models. International
  Journal of Climatology 31(10), 1530-1544.
- Van Wagner C.E. (1987). Development and structure of the Canadian Forest Fire
  Weather Index System. Canadian Forestry Service Technical Report 35.
- Duffie J.A. et al. (2020). Solar Engineering of Thermal Processes,
  Photovoltaics and Wind. 5th ed. Wiley.

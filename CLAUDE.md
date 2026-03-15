# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demand forecasting and safety stock planning system implemented as a single Jupyter notebook (`Forecast_TimesFM_and_SS.ipynb`) designed to run in **Google Colab**. It combines Google's TimesFM-2.5-200M deep learning model with classical inventory optimization (ABC/XYZ classification and safety stock calculations).

## Running the Notebook

There is no traditional build system. The notebook is executed sequentially in Google Colab:

1. Upload the notebook to Colab and run cells top-to-bottom
2. Cell 15 installs dependencies via inline `!pip install` commands (TimesFM, torch, einops, huggingface_hub)
3. TimesFM model is auto-downloaded from HuggingFace (`google/timesfm-2.5-200m-pytorch`)
4. Input: Excel file uploaded via `google.colab.files.upload()`
5. Output: `Forecast and SS v0.xlsx` exported via `google.colab.files.download()`

## Architecture: 11 Functional Modules (A–K)

All code lives in a single notebook with cells organized into labeled modules:

- **Module A** — Global parameters and configuration constants (edit here to change behavior)
- **Module B** — Data loading: reads Excel, detects `YYYY_MM` temporal columns, converts wide→long, filters by minimum history, winsorizes outliers
- **Module C** — Time series construction; builds backtest dataset (truncated history + held-out actuals)
- **Module D** — Seasonal calibration using Theil-Sen log-linear regression; per-SKU factors with global fallback. Defines `theil_sen_log_trend()` — the single canonical implementation used by both Module D and Module G
- **Module E** — `round_to_pack()`: pack-size-aware rounding with modes `"up"/"down"/"nearest"`
- **Module F** — Manual TimesFM model loader (Python 3.12 Colab-compatible); GPU/CPU auto-detection; smoke test
- **Module G** — Backtest engine: grid search over scaling factors (0.10–0.90), per-SKU, no data leakage; outputs best scaling factor per SKU. Uses `theil_sen_log_trend()` from Module D
- **Module H** — Future forecast generation (applies best scaling factor + calibration + pack rounding); outputs wide-format forecast table
- **Modules I–J** — ABC classification (Pareto on volume), XYZ classification (CV thresholds), safety stock (`Z * σ * √((LT + ReorderPeriod)/30)`)
- **Module K** — Final Excel export merging metadata, historical demand, forecasts, and inventory metrics

## Key Configuration (Module A)

| Parameter | Default | Purpose |
|---|---|---|
| `HORIZON` | 25 | Months to forecast |
| `HORIZON_BACKTEST` | 12 | Backtest evaluation window |
| `MIN_HISTORY_POINTS` | 6 | Minimum historical months per SKU |
| `REMOVE_OUTLIERS` | True | Enable winsorizing |
| `OUTLIER_LEVEL` | 0.05 | Clip at 5th/95th percentile |
| `CALIBRATION_MONTHS` | [8, 12] | Months with seasonal adjustment (August, December) |
| `DEFAULT_LEAD_TIME` | 30 | Fallback lead time in days |
| `SS_LOOKBACK_MONTHS` | 12 | Lookback window for σ in safety stock |
| `ROUNDING_MODE` | `"nearest"` | Pack rounding mode |

Column mappings (SKU ID, description, lead time, pack size, UoM) are also defined in Module A — update these to match a new Excel schema.

## Important Design Decisions

- **No data leakage**: backtest series are strictly truncated before the evaluation window
- **Leading zeros only are trimmed**: zeros at the start of a series (product not yet launched) are removed; internal and trailing zeros are kept as real demand observations
- **Single canonical Theil-Sen**: `theil_sen_log_trend()` defined in Module D uses full all-pairs Theil-Sen on the complete series (including internal zeros, preserving actual temporal positions). Used identically in both calibration (Module D) and backtest (Module G) to ensure consistency
- **Bidirectional calibration**: seasonal factors can both increase and decrease forecasts
- **Scaling factor optimized via backtest**: the `QUANTILE_GRID` search finds the multiplicative scaling factor (`q / 0.5`) that maximises Motul weighted accuracy on held-out data — not TimesFM's native quantile outputs, which optimise for pinball loss instead
- **Volume-weighted accuracy**: `ACC = Σ(ACC_i × (ACT_i + FCST_i)) / Σ(ACT_i + FCST_i)`, not simple mean
- **Safety stock rounded UP** to pack multiples regardless of `ROUNDING_MODE`
- **Service levels by ABC/XYZ class matrix**: AX=97%, CZ=0% (no safety stock for low-value/erratic)
- **ABC guard**: if total volume in the lookback window is zero, all SKUs default to class C to avoid division by zero

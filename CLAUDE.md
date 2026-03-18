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

## Architecture: 10 Functional Modules (A–J)

All code lives in a single notebook with cells organized into labeled modules:

- **Module A** — Global parameters and configuration constants (edit here to change behavior)
- **Module B** — Data loading: reads Excel, detects `YYYY_MM` temporal columns, converts wide→long, filters by minimum history, winsorizes outliers
- **Module C** — Time series construction; builds backtest dataset (truncated history + held-out actuals); defines `accuracy_single_month()` and `accuracy_weighted()` (Motul formula)
- **Module D** — Seasonal calibration using Theil-Sen log-linear regression; per-SKU factors with global fallback. Defines `theil_sen_log_trend()` — the single canonical implementation (vectorized with numpy) used by both Module D and Module G
- **Module E** — `round_to_pack()`: pack-size-aware rounding with modes `"up"/"down"/"nearest"`
- **Module F** — Manual TimesFM model loader (Python 3.12 Colab-compatible); GPU/CPU auto-detection; smoke test. Forecast function uses batch inference with automatic fallback to per-SKU if batch fails
- **Module G** — Backtest engine: rolling-origin grid search over scaling factors, per-SKU, no data leakage. Two-pass grid: coarse (step 0.05) then fine (step 0.01) around the best. Supports multiple backtest origins (`N_BACKTEST_ORIGINS`) for robustness and optional shrinkage of per-SKU scaling factors toward global median (`SHRINKAGE_ENABLED`). Uses batch forecast with automatic per-SKU fallback. Uses `theil_sen_log_trend()` from Module D
- **Module H** — Future forecast generation (applies best scaling factor + calibration + pack rounding); outputs wide-format forecast table
- **Module I** — ABC classification (Pareto on volume), XYZ classification (CV thresholds), safety stock (`Z * σ * √((LT + ReorderPeriod)/30)`)
- **Module J** — Final Excel export merging metadata, historical demand, forecasts, and inventory metrics

## Key Configuration (Module A)

| Parameter | Default | Purpose |
|---|---|---|
| `HORIZON` | 25 | Months to forecast |
| `HORIZON_BACKTEST` | 12 | Backtest evaluation window |
| `MIN_HISTORY_POINTS` | 6 | Minimum historical months per SKU |
| `REMOVE_OUTLIERS` | True | Enable winsorizing |
| `OUTLIER_LEVEL` | 0.05 | Clip at 5th/95th percentile |
| `CALIBRATION_MONTHS` | [8, 12] | Months with seasonal adjustment (August, December); `[]` to disable |
| `DEFAULT_LEAD_TIME` | 30 | Fallback lead time in days (used if `LT` column is missing) |
| `REORDER_PERIOD` | 30 | Review period in days (fixed at 1 month per business requirement) |
| `SS_LOOKBACK_MONTHS` | 12 | Lookback window for σ in safety stock |
| `CALCULATE_SS` | True | Enable/disable safety stock calculation |
| `TRIM_LEADING_ZEROS` | True | Remove leading zeros (pre-launch periods) from each series |
| `QUANTILE_GRID` | 0.10–0.90 step 0.05 | Scaling factor search grid (17 points); a fine-grid refinement (step 0.01) runs automatically around the best coarse result |
| `N_BACKTEST_ORIGINS` | 2 | Number of rolling backtest origins (1 = single split, 2+ = rolling-origin cross-validation with 6-month shift between origins) |
| `SHRINKAGE_ENABLED` | True | Blend per-SKU scaling factor with global median; trust weight (α) scales linearly with history length up to 36 months |
| `ROUNDING_MODE` | `"nearest"` | Pack rounding mode for forecasts (`"up"` / `"down"` / `"nearest"`) |
| `ROUND_DECIMALS` | 3 | Decimal places in rounded output values |

Column mappings (defaults match the Motul Excel schema — update if input file has different column names):

| Variable | Default | Excel column |
|---|---|---|
| `ID_COL` | `"SKU"` | Product code (unique key) |
| `DESC_COL` | `"Description"` | Product description |
| `LT_COL_NAME` | `"LT"` | Lead time in days |
| `PACK_SIZE_COL` | `"Round"` | Pack size / order multiple |
| `UOM_COL` | `"BUn"` | Unit of measure |

## Important Design Decisions

- **No data leakage**: backtest series are strictly truncated before the evaluation window
- **Leading zeros only are trimmed**: zeros at the start of a series (product not yet launched) are removed; internal and trailing zeros are kept as real demand observations
- **Single canonical Theil-Sen**: `theil_sen_log_trend()` defined in Module D uses full all-pairs Theil-Sen on the complete series (including internal zeros, preserving actual temporal positions). Used identically in both calibration (Module D) and backtest (Module G) to ensure consistency
- **Bidirectional calibration**: seasonal factors can both increase and decrease forecasts
- **Scaling factor optimized via backtest**: two-pass grid search (coarse step 0.05, then fine step 0.01 around the best) finds the multiplicative scaling factor (`q / 0.5`) that maximises Motul weighted accuracy on held-out data — not TimesFM's native quantile outputs, which optimise for pinball loss instead. When `N_BACKTEST_ORIGINS > 1`, accuracy is averaged across multiple rolling origins (shifted by 6 months each) for robustness. Optional shrinkage blends per-SKU optimal q toward the global median, weighted by history length (full trust at ≥ 36 months)
- **Motul accuracy formula (fixed business requirement)**: `ACC_i = 1 - |ACT - FCST| / ACT`, but returns **0** if ACT ≤ 0, FCST ≤ 0, FCST < ACT/2 (under-forecast by more than half), or FCST > 2×ACT (over-forecast by more than double). Do NOT modify — defined by Casa Madre. The entire backtest and scaling-factor optimization exists to maximize this metric.
- **Volume-weighted accuracy**: `ACC = Σ(ACC_i × (ACT_i + FCST_i)) / Σ(ACT_i + FCST_i)`, not simple mean
- **Safety stock rounded UP** to pack multiples regardless of `ROUNDING_MODE`
- **Service levels by ABC/XYZ class matrix**: AX=97%, CZ=0% (no safety stock for low-value/erratic)
- **ABC guard**: if total volume in the lookback window is zero, all SKUs default to class C to avoid division by zero

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demand forecasting and safety stock planning system designed to run in **Google Colab** or **locally** on any machine with Python 3.10+. The execution mode is controlled by the `COLAB` variable in Module A. It combines Google's TimesFM-2.5-200M deep learning model with classical inventory optimization (ABC/XYZ classification and safety stock calculations).

Since v1.5.0 the project is split in two layers:
- **Notebook** (`Forecast_TimesFM_and_SS.ipynb`) — configuration (Module A) and orchestration only.
- **Package `forecast_lib/`** — all pipeline math, in plain `.py` files (one per module). In Colab the package is `git clone`d from GitHub at notebook startup.

## Running the Notebook

There is no traditional build system. The notebook supports two execution modes controlled by `COLAB` in Module A:

**Google Colab (`COLAB = True`, default):**
1. Open the notebook in Colab and run cells top-to-bottom
2. The bootstrap cell (Module A.2) clones the GitHub repo to `/content/Forecast-TimesFM-and-SS/` and adds it to `sys.path`, then imports `forecast_lib`. Each new Colab session always pulls the latest `main`.
3. The Module F cell installs TimesFM dependencies via inline `!pip install` (torch, einops, huggingface_hub)
4. TimesFM model is auto-downloaded from HuggingFace (`google/timesfm-2.5-200m-pytorch`)
5. Input: Excel file uploaded via `google.colab.files.upload()`
6. Output: Excel file exported via `google.colab.files.download()`

**Local execution (`COLAB = False`):**
1. Dependencies must be pre-installed via `pip install -r requirements.txt` (CPU) or `pip install -r requirements-nvidia.txt` (NVIDIA GPU)
2. The bootstrap cell adds `os.getcwd()` to `sys.path` (notebook lives next to `forecast_lib/`) and imports the package
3. TimesFM source repo is auto-cloned to `./timesfm/` if not already present
4. HuggingFace model uses default cache (`~/.cache/huggingface`) with automatic ETag-based update checks
5. Input: Excel file selected via `tkinter.filedialog` (works in classic Jupyter Notebook; may require manual path in JupyterLab/VS Code)
6. Output: saved to `./output/` directory, or user-chosen path if `ASK_SAVE_PATH = True`

## Project Layout

```
Forecast_TimesFM_and_SS.ipynb   # 13 cells: Module A config, bootstrap+imports, then one cell per pipeline stage
forecast_lib/
  __init__.py
  preprocessing.py              # Module B helpers
  metrics.py                    # Module C: accuracy_single_month, accuracy_weighted (Motul formula)
  calibration.py                # Module D: theil_sen_log_trend, compute_calibration_factors, get_calibration_factor, calculate_seasonality_local
  rounding.py                   # Module E: round_to_pack
  model.py                      # Module F: setup_timesfm, forecast_all_skus_point
  backtest.py                   # Module G: run_backtest, empty_backtest_results
  inventory.py                  # Module I: calculate_inventory_logic
  export.py                     # Module J: build_forecast_wide, build_final_table, save_excel
tests/                          # pytest suite (run with `pytest` from repo root)
requirements.txt / -nvidia.txt  # CPU / GPU dependency pins
```

## Architecture: notebook + `forecast_lib/`

The notebook is organized in 13 cells (down from 23 pre-v1.5.0): one for Module A config, one for the bootstrap+imports, then one cell per logical pipeline stage that calls into `forecast_lib`. Module → file mapping:

- **Module A** *(notebook, first cell)* — Global parameters and configuration constants (edit here to change behavior)
- **Module B** → `forecast_lib/preprocessing.py` — Data loading helpers: `wide_to_long` (detect `YYYY_MM` columns, NaN→0, melt, parse dates), `filter_min_history`, `apply_winsorize`, `winsorize_series`, `build_sku_series` (trim leading zeros), `build_backtest_series` (truncate for backtest)
- **Module C** → `forecast_lib/metrics.py` — Motul accuracy formula: `accuracy_single_month()`, `accuracy_weighted()`. Build of backtest dataset stays alongside Module B helpers (`build_backtest_series`).
- **Module D** → `forecast_lib/calibration.py` — `theil_sen_log_trend()` (single canonical implementation, used by both calibration and backtest), `compute_calibration_factors()` (per-SKU + global), `get_calibration_factor()` (lookup), `calculate_seasonality_local()` (no-leakage version used inside the backtest)
- **Module E** → `forecast_lib/rounding.py` — `round_to_pack()` with modes `"up"/"down"/"nearest"`
- **Module F** → `forecast_lib/model.py` — `setup_timesfm()` (manual loader, Python 3.12 Colab-compatible, GPU/CPU auto-detection, smoke test) and `forecast_all_skus_point()` (batch inference with automatic per-SKU fallback)
- **Module G** → `forecast_lib/backtest.py` — `run_backtest()`: rolling-origin grid search over scaling factors, no data leakage. Two-pass grid: coarse (step 0.05) then fine (step 0.01) around the best. Supports multiple origins (`N_BACKTEST_ORIGINS`) for robustness and optional shrinkage toward global median (`SHRINKAGE_ENABLED`). Uses `theil_sen_log_trend()` and `calculate_seasonality_local()` from `calibration.py`. When `RUN_BACKTEST = False`, the notebook calls `empty_backtest_results()` and Module H falls back to `q = 0.5` for all SKUs
- **Module H** *(notebook, with helpers from `calibration.py` and `rounding.py`)* — Future forecast generation: applies best scaling factor + calibration + business adjustment + pack rounding. The business adjustment (`BUSINESS_ADJUSTMENT_FACTOR`) is a manual multiplier applied between calibration and rounding, intended as a managerial procurement lever orthogonal to the model
- **Module I** → `forecast_lib/inventory.py` — `calculate_inventory_logic()`: ABC (Pareto on volume), XYZ (CV thresholds), safety stock (`Z * σ * √((LT + ReorderPeriod)/30)`)
- **Module J** → `forecast_lib/export.py` — `build_forecast_wide()`, `build_final_table()`, `save_excel()`. The Colab download / local save dialog logic stays in the notebook (environment-specific I/O).

### Tests

`tests/` contains pytest tests for the pure functions in `forecast_lib/`. Run with `pytest` from the repo root. Coverage:
- `test_metrics.py` — Motul formula edge cases (4 zero-cases, weighted formula)
- `test_rounding.py` — three rounding modes + edge cases (NaN, pack=0/None/negative, integer-multiple values)
- `test_calibration.py` — Theil-Sen on pure exponential, constant series, internal-zeros, robustness to outliers; `get_calibration_factor` priority logic
- `test_preprocessing.py` — wide→long, min-history filter, leading/internal/trailing zero handling, backtest split logic
- `test_inventory.py` — ABC/XYZ classification, ABC zero-volume guard, CZ→SS=0, SS rounded up to pack, LT column override
- `test_export.py` — final table merge, forecast column prefix, missing-SS fill

## Key Configuration (Module A)

| Parameter | Default | Purpose |
|---|---|---|
| `COLAB` | True | Execution mode: `True` = Google Colab, `False` = local execution |
| `ASK_SAVE_PATH` | False | Local mode only: `True` = open save dialog for output file, `False` = save to `./output/` |
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
| `RUN_BACKTEST` | True | Master switch for Module G. `False` skips the entire backtest; all SKUs use `q = 0.5` (TimesFM native median, **not** optimized for the Motul KPI) |
| `SHRINKAGE_ENABLED` | True | Blend per-SKU scaling factor with global median; trust weight (α) scales linearly with history length up to 36 months. Has effect only when `RUN_BACKTEST = True` |
| `BUSINESS_ADJUSTMENT_FACTOR` | 1.0 | Multiplier applied to the forecast in Module H, between seasonal calibration and pack rounding. Managerial procurement lever, orthogonal to the model: `<1.0` lowers the forecast, `>1.0` raises it. `1.0` is the neutral default |
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
- **Single canonical Theil-Sen**: `theil_sen_log_trend()` lives in `forecast_lib/calibration.py` and uses full all-pairs Theil-Sen on the complete series (including internal zeros, preserving actual temporal positions). Used identically by `compute_calibration_factors()` (production calibration) and by `calculate_seasonality_local()` (no-leakage variant called inside `backtest.run_backtest`). Modify this function in one place only
- **Bidirectional calibration**: seasonal factors can both increase and decrease forecasts
- **Scaling factor optimized via backtest**: two-pass grid search (coarse step 0.05, then fine step 0.01 around the best) finds the multiplicative scaling factor (`q / 0.5`) that maximises Motul weighted accuracy on held-out data — not TimesFM's native quantile outputs, which optimise for pinball loss instead. When `N_BACKTEST_ORIGINS > 1`, accuracy is averaged across multiple rolling origins (shifted by 6 months each) for robustness. Optional shrinkage blends per-SKU optimal q toward the global median, weighted by history length (full trust at ≥ 36 months)
- **Backtest is the only Motul-aware step**: `accuracy_weighted()` is invoked exclusively inside `backtest.run_backtest`. Setting `RUN_BACKTEST = False` removes all Motul-driven optimization (the notebook calls `empty_backtest_results()` and the forecast falls back to `q = 0.5` for every SKU). Use only for fast simulations or methodological A/B tests, never as the production default
- **Business adjustment is post-model and orthogonal**: `BUSINESS_ADJUSTMENT_FACTOR` multiplies the forecast in the notebook's Module H cell *after* backtest scaling and seasonal calibration but *before* pack rounding. It is intended for procurement scenario adjustments (crisis, market shifts, stock constraints), not for tuning the model. Keeping it separate makes its impact auditable and prevents conflating model accuracy with managerial choice. Note: with `ROUNDING_MODE = "up"` and a factor `< 1`, the rounding step can absorb part of the reduction on SKUs with large packs — expected behavior, consistent with procurement logic
- **Motul accuracy formula (fixed business requirement)**: `ACC_i = 1 - |ACT - FCST| / ACT`, but returns **0** if ACT ≤ 0, FCST ≤ 0, FCST < ACT/2 (under-forecast by more than half), or FCST > 2×ACT (over-forecast by more than double). Do NOT modify `forecast_lib/metrics.py` — defined by Casa Madre. The entire backtest and scaling-factor optimization exists to maximize this metric. `tests/test_metrics.py` pins this behavior with explicit cases.
- **Volume-weighted accuracy**: `ACC = Σ(ACC_i × (ACT_i + FCST_i)) / Σ(ACT_i + FCST_i)`, not simple mean
- **Safety stock rounded UP** to pack multiples regardless of `ROUNDING_MODE`
- **Service levels by ABC/XYZ class matrix**: AX=97%, CZ=0% (no safety stock for low-value/erratic)
- **ABC guard**: if total volume in the lookback window is zero, all SKUs default to class C to avoid division by zero

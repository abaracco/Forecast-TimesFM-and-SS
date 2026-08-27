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
3. The Module F cell installs TimesFM dependencies via inline `!pip install` (torch, safetensors, huggingface_hub)
4. TimesFM source is cloned to `/content/timesfm` **pinned to the `TIMESFM_VERSION` tag and verified**; the model weights are downloaded from HuggingFace (`google/timesfm-2.5-200m-pytorch`) **at the pinned `TIMESFM_MODEL_REVISION`**
5. Input: Excel file uploaded via `google.colab.files.upload()`
6. Output: Excel file exported via `google.colab.files.download()`

**Local execution (`COLAB = False`):**
1. Dependencies must be pre-installed via `pip install -r requirements.txt` (CPU) or `pip install -r requirements-nvidia.txt` (NVIDIA GPU)
2. The bootstrap cell adds `os.getcwd()` to `sys.path` (notebook lives next to `forecast_lib/`) and imports the package
3. TimesFM source repo is cloned to `./timesfm/` pinned to the `TIMESFM_VERSION` tag; an existing checkout is verified and re-cloned (clone-then-swap) if it drifted
4. HuggingFace model uses default cache (`~/.cache/huggingface`) at the pinned revision — which also means it works offline after the first download
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
  export.py                     # Module J: build_forecast_wide, build_final_table, save_excel, build_run_info, save_audit_csvs
  versioning.py                 # outside the module numbering: TimesFM pin + update checks
tests/                          # pytest suite (`pytest`, plus `pytest -m slow`)
  _fake_timesfm.py              # TimesFM double used by the fast tests (not collected)
  tools/compare_forecast_outputs.py  # two-run comparison: gates G1-G5 + diagnostics
requirements.txt / -nvidia.txt  # CPU / GPU dependency pins
```

## Architecture: notebook + `forecast_lib/`

The notebook is organized in 13 cells (down from 23 pre-v1.5.0): one for Module A config, one for the bootstrap+imports, then one cell per logical pipeline stage that calls into `forecast_lib`. Module → file mapping:

- **Module A** *(notebook, first cell)* — Global parameters and configuration constants (edit here to change behavior)
- **Module B** → `forecast_lib/preprocessing.py` — Data loading helpers: `wide_to_long` (detect `YYYY_MM` columns, NaN→0, melt, parse dates), `filter_min_history`, `apply_winsorize`, `winsorize_series`, `build_sku_series` (trim leading zeros), `build_backtest_series` (truncate for backtest)
- **Module C** → `forecast_lib/metrics.py` — Motul accuracy formula: `accuracy_single_month()`, `accuracy_weighted()`. Build of backtest dataset stays alongside Module B helpers (`build_backtest_series`).
- **Module D** → `forecast_lib/calibration.py` — `theil_sen_log_trend()` (single canonical implementation, used by both calibration and backtest), `compute_calibration_factors()` (per-SKU + global), `get_calibration_factor()` (lookup), `calculate_seasonality_local()` (no-leakage version used inside the backtest)
- **Module E** → `forecast_lib/rounding.py` — `round_to_pack()` with modes `"up"/"down"/"nearest"`
- **Module F** → `forecast_lib/model.py` — `setup_timesfm()` (explicit loader: pinned+verified TimesFM checkout via `versioning.py`, explicit module path and model class name, weights at a pinned revision, `per_core_batch_size=INFERENCE_BATCH_SIZE`, GPU/CPU auto-detection, **blocking** smoke test), `forecast_batch_with_fallback()` (the single entry point to inference: batch attempt → OOM degradation along `[N, N//4, 1]` → per-input loop at batch 1) and `forecast_all_skus_point()`. Run state travels as `fl_*` attributes on the model instance (`fl_batch_size`, `fl_batch_size_initial`, `fl_degraded`, `fl_degraded_after_inference`, `fl_has_produced_output`, `fl_device`, `fl_model_revision`, `fl_inference_seconds`, `fl_pin_verified`, `fl_timesfm_tag`) plus the `fl_recompile()` closure — the signature still returns just the model, so no call site changed
- **`versioning.py`** *(no module letter)* — `ensure_timesfm_checkout()` (the only function here that may raise, and only when `strict=True`), `timesfm_tag()`, `compare_versions()`, `latest_timesfm_tag()`, `latest_lib_version()`, `check_library_version()`, `local_repo_status()`, and the two non-blocking orchestrators `check_project_updates()` (cell 1) and `check_timesfm_update()` (cell 6)
- **Module G** → `forecast_lib/backtest.py` — `run_backtest()`: rolling-origin grid search over scaling factors, no data leakage. Two-pass grid: coarse (step 0.05) then fine (step 0.01) around the best. Supports multiple origins (`N_BACKTEST_ORIGINS`) for robustness and optional shrinkage toward global median (`SHRINKAGE_ENABLED`). Uses `theil_sen_log_trend()` and `calculate_seasonality_local()` from `calibration.py`. When `RUN_BACKTEST = False`, the notebook calls `empty_backtest_results()` and Module H falls back to `q = 0.5` for all SKUs. The result frame also carries `q_global`, `n_backtest_skus`, `n_skus_excluded` and `n_skus_zero_accuracy` in `df.attrs` (and `q_global` as a constant CSV column, since `attrs` does not survive `to_csv`), plus `BestQuantileRaw`/`BestAccuracyRaw` — the pre-shrinkage values, without which an individual flip cannot be told apart from a shift of the global median
- **Module H** *(notebook, with helpers from `calibration.py` and `rounding.py`)* — Future forecast generation: applies best scaling factor + calibration + business adjustment + pack rounding. The business adjustment (`BUSINESS_ADJUSTMENT_FACTOR`) is a manual multiplier applied between calibration and rounding, intended as a managerial procurement lever orthogonal to the model
- **Module I** → `forecast_lib/inventory.py` — `calculate_inventory_logic()`: ABC (Pareto on volume), XYZ (CV thresholds), safety stock (`Z * σ * √((LT + ReorderPeriod)/30)`)
- **Module J** → `forecast_lib/export.py` — `build_forecast_wide()`, `build_final_table()`, `save_excel(df, path, run_info=None)`, `build_run_info()`, `run_info_to_frame()`, `save_audit_csvs()`. The Colab download / local save dialog logic stays in the notebook (environment-specific I/O). **The data table is always the first sheet**, even when the "Run info" sheet is present — this project reads its own input with `list(all_sheets.keys())[0]`.

### Tests

`tests/` contains the pytest suite. Run `pytest` from the repo root for the fast, offline suite; `pytest -m slow` for the tests that need the network or the real TimesFM model (`pytest.ini` sets `addopts = -m "not slow"`, which a command-line `-m` overrides). Coverage:
- `test_metrics.py` — Motul formula edge cases (4 zero-cases, weighted formula)
- `test_rounding.py` — three rounding modes + edge cases (NaN, pack=0/None/negative, integer-multiple values)
- `test_calibration.py` — Theil-Sen on pure exponential, constant series, internal-zeros, robustness to outliers; `get_calibration_factor` priority logic
- `test_preprocessing.py` — wide→long, min-history filter, leading/internal/trailing zero handling, backtest split logic
- `test_inventory.py` — ABC/XYZ classification, ABC zero-volume guard, CZ→SS=0, SS rounded up to pack, LT column override
- `test_export.py` — final table merge, forecast column prefix, missing-SS fill, `save_excel` sheet layout (one sheet / two sheets, data always first, readable without `sheet_name`), `build_run_info`
- `test_versioning.py` (T2) — version comparison, `ls-remote` tag parsing (`^{}` dedup), the two distinct `check_library_version` messages, orchestrator guards with `subprocess` mocked
- `test_model_config.py` (T3) — `setup_timesfm` run end-to-end against a TimesFM double that is actually executed by `spec_from_file_location`. Assertions are on the **arguments passed to the `ForecastConfig` constructor**, never on model state: `compile()` rewrites `max_horizon` to 128. Covers the whole degradation group (OOM scale, non-OOM path, smoke-test isolation, no caller-list mutation) and `run_backtest`'s `q_global`
- `test_compare_forecast_outputs.py` — every gate of the comparison utility fires when it should and only when it should
- `test_versioning_integration.py` (T2b, `slow`) — `ensure_timesfm_checkout` against real git repos: one real clone of `google-research/timesfm` (the LFS canary), everything else against a local `git init` remote so the degraded cases are reproducible on demand. Also `local_repo_status`, `latest_lib_version` and the notebook's branch-alignment sequence
- `test_model_integration.py` (T1, `slow`) — on the real model: batch-32 inference matches batch-1 (`rtol=1e-4`) before and after pack rounding, padding does not contaminate the last chunk, output cardinality, no input mutation

`tests/tools/compare_forecast_outputs.py` is not a test: it is the utility that scores two runs against gates G1-G5 (refactor identity, structural identity, aggregate impact with and without sign, explainability) and prints the diagnostics. Use it whenever a change could move the numbers — see the "Aggiornare TimesFM" runbook in the README.

## Key Configuration (Module A)

| Parameter | Default | Purpose |
|---|---|---|
| `COLAB` | True | Execution mode: `True` = Google Colab, `False` = local execution |
| `TIMESFM_VERSION` | `"2.0.2"` | TimesFM source version, **without the leading `v`**. The git tag is `f"v{TIMESFM_VERSION}"`; the `v` is added in exactly one place, `versioning.timesfm_tag()` |
| `TIMESFM_REPO_URL` | official repo | TimesFM repository. If `./timesfm` exists with a different `origin`, the loader raises instead of touching it |
| `TIMESFM_PIN_STRICT` | True | `True` = an unverifiable pin blocks the run. `False` = proceed with a loud warning; the run is then flagged in "Run info" and in an end-of-run warning |
| `TIMESFM_MODEL_ID` | `"google/timesfm-2.5-200m-pytorch"` | HuggingFace model |
| `TIMESFM_MODEL_REVISION` | commit hash | Weights revision, **pinned on purpose**. Unpinned, HuggingFace resolves `main` and a Google-side weight update would change every forecast silently. Being pinned it does not update itself: re-evaluate it on every `TIMESFM_VERSION` change |
| `INFERENCE_BATCH_SIZE` | 32 | Series sent to the model per pass, per device (`per_core_batch_size`). The TimesFM default is 1. On OOM the code degrades along `[N, N//4, 1]` by itself |
| `EXPECTED_FORECAST_LIB_VERSION` | `"1.6.0"` | Version of `forecast_lib` this notebook expects; must match `forecast_lib.__version__` |
| `REPO_BRANCH` | `"main"` | Branch cloned in Colab. Keep on `main` in production; change **only** to test a working branch |
| `CHECK_FOR_UPDATES` | True | Startup check for newer TimesFM / `forecast_lib` versions. Prints a warning, never updates anything. Active in Colab too — that is where a pin ages unnoticed |
| `EXPORT_AUDIT` | True | Emit the second "Run info" sheet and the two audit CSVs next to the output |
| `ASK_SAVE_PATH` | False | Local mode only: `True` = open save dialog for output file, `False` = save to `./output/` |
| `HORIZON` | 24 | Months to forecast (2 years) |
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

- **Both the code and the weights are pinned, and the pin is verified**: `TIMESFM_VERSION` fixes the source tag, `TIMESFM_MODEL_REVISION` the weights commit. Verification is `HEAD == refs/tags/<tag>^{commit}` **and** a clean working tree — a pin that is declared but never checked gives false confidence, which is worse than none. `TIMESFM_PIN_STRICT = False` is the deliberate escape hatch, and it leaves a trace in the output file
- **Sparse clone because of the LFS filters**: TimesFM is cloned with `--filter=blob:none --sparse` + `sparse-checkout set src`. Verified over 15 clones as the only variant whose working tree stays reliably clean — the files subject to the repo's LFS filters simply do not exist in the checkout. `GIT_LFS_SKIP_SMUDGE` is not enough (it acts on smudge, not on clean). Without this, the "clean tree" check would fail at random and pin verification would be unusable
- **Clone-then-swap when updating the checkout**: the existing folder is never deleted before a successful clone, the temp dir is created *next to* the destination (a cross-volume swap fails on Windows), and `rmtree` uses a `chmod` handler because `.git/objects/pack/*.idx|.pack` are read-only. A different `origin` raises without touching anything: `./timesfm` is relative to the CWD, which in JupyterLab/VS Code may not be the project root
- **OOM degradation, and the run that must not be delivered**: on OOM the batch size steps down `[N, N//4, 1]` (falling straight to 1 throws away ~40x when an intermediate level often suffices); on any other failure it drops to batch 1 and loops per input. Degradation is permanent for the run but **does not make the run coherent** — whatever was already computed stays at the previous batch size. Hence two distinct flags: `fl_degraded` (it happened) and `fl_degraded_after_inference` (**the run is not deliverable**, redo it with a lower `INFERENCE_BATCH_SIZE`). The second one is raised only when at least one real inference had already succeeded at a different batch size (`fl_has_produced_output`) — that is exactly the case where the run's numbers were computed at mixed batch sizes. A degradation during the smoke test, or on the very first attempt of the first real call, leaves the run uniform at the lower batch size and fully usable: flagging it would force a pointless multi-minute rerun
- **`pip install timesfm[torch]` evaluated and rejected**: beyond Python 3.12 compatibility in Colab, a pip install offers no equally direct way to verify *which* code is running. A git tag with a comparable `HEAD` does
- **Blocking smoke test**: after loading, the model is queried through the pipeline's own inference path and the output is checked for shape and finiteness, not merely for the absence of an exception. With a batch size above 1 that single series immediately exercises TimesFM's internal padding branch, so it doubles as a canary
- **Scaling factor on zero-accuracy SKUs — known limitation, out of scope**: the Motul formula returns 0 whenever the forecast is below half or above twice the actual, which for erratic SKUs can happen across the *whole* grid. There `max()` returns the first key of `QUANTILE_GRID` (`q = 0.10`), so the chosen scaling factor is an artefact of iteration order, not a model decision. These SKUs are **not** confined to class C — zero accuracy follows erraticity (XYZ), not volume (ABC), and an A/Z SKU is the typical candidate. Counted as `n_skus_zero_accuracy` and reported in "Run info"; fixing it deserves its own cycle
- **No data leakage**: backtest series are strictly truncated before the evaluation window
- **Leading zeros only are trimmed**: zeros at the start of a series (product not yet launched) are removed; internal and trailing zeros are kept as real demand observations
- **Single canonical Theil-Sen**: `theil_sen_log_trend()` lives in `forecast_lib/calibration.py` and uses full all-pairs Theil-Sen on the complete series (including internal zeros, preserving actual temporal positions). Used identically by `compute_calibration_factors()` (production calibration) and by `calculate_seasonality_local()` (no-leakage variant called inside `backtest.run_backtest`). Modify this function in one place only
- **Bidirectional calibration**: seasonal factors can both increase and decrease forecasts
- **Scaling factor optimized via backtest**: two-pass grid search (coarse step 0.05, then fine step 0.01 around the best) finds the multiplicative scaling factor (`q / 0.5`) that maximises Motul weighted accuracy on held-out data — not TimesFM's native quantile outputs, which optimise for pinball loss instead. When `N_BACKTEST_ORIGINS > 1`, accuracy is averaged across multiple rolling origins (shifted by 6 months each) for robustness. Optional shrinkage blends per-SKU optimal q toward the global median, weighted by history length (full trust at ≥ 36 months)
- **Backtest is the only Motul-aware step**: `accuracy_weighted()` is invoked exclusively inside `backtest.run_backtest`. Setting `RUN_BACKTEST = False` removes all Motul-driven optimization (the notebook calls `empty_backtest_results()` and the forecast falls back to `q = 0.5` for every SKU). Use only for fast simulations or methodological A/B tests, never as the production default
- **Business adjustment is post-model and orthogonal**: `BUSINESS_ADJUSTMENT_FACTOR` multiplies the forecast in the notebook's Module H cell *after* backtest scaling and seasonal calibration but *before* pack rounding. It is intended for procurement scenario adjustments (crisis, market shifts, stock constraints), not for tuning the model. Keeping it separate makes its impact auditable and prevents conflating model accuracy with managerial choice. Note: with `ROUNDING_MODE = "up"` and a factor `< 1`, the rounding step can absorb part of the reduction on SKUs with large packs — expected behavior, consistent with procurement logic
- **Motul accuracy formula (fixed business requirement)**: `ACC_i = 1 - |ACT - FCST| / ACT`, but returns **0** if ACT ≤ 0, FCST ≤ 0, FCST < ACT/2 (under-forecast by more than half), or FCST > 2×ACT (over-forecast by more than double). Do NOT modify `forecast_lib/metrics.py` — fixed business requirement. The entire backtest and scaling-factor optimization exists to maximize this metric. `tests/test_metrics.py` pins this behavior with explicit cases.
- **Volume-weighted accuracy**: `ACC = Σ(ACC_i × (ACT_i + FCST_i)) / Σ(ACT_i + FCST_i)`, not simple mean
- **Safety stock rounded UP** to pack multiples regardless of `ROUNDING_MODE`
- **Service levels by ABC/XYZ class matrix**: AX=97%, CZ=0% (no safety stock for low-value/erratic)
- **ABC guard**: if total volume in the lookback window is zero, all SKUs default to class C to avoid division by zero

## Decision log — TimesFM pin + batch inference (v1.6.0, 2026-08-27)

Carried over from the working plan so it survives the plan file. These are the
choices a future change should not silently undo.

| Decision | Outcome / rationale |
|---|---|
| Pinned clone vs `pip install timesfm[torch]` | **Pinned clone at a git tag.** A pip install gives no equally direct way to verify which code is running |
| How to clone TimesFM | **`--filter=blob:none --sparse` + `sparse-checkout set src`.** Verified over 15 clones: the only variant with a reliably clean tree. `GIT_LFS_SKIP_SMUDGE` is not enough — it acts on smudge, not on clean |
| How to update the clone | **Clone into a temp dir next to the destination, then swap**, after checking the remote, with a `chmod` handler for Windows read-only pack files. Never delete first: if the clone fails because the network is down — the very scenario `TIMESFM_PIN_STRICT = False` exists for — the old folder must still be usable |
| `TIMESFM_MODEL_REVISION` | **Pinned.** Zero cost, and the "Aggiornare TimesFM" runbook re-evaluates it on every version change. Also improves offline use: HuggingFace reuses the cache instead of re-resolving `main` |
| Pin verification | **Blocking by default**, with `TIMESFM_PIN_STRICT = False` as the escape hatch, traced in "Run info" and in an end-of-run warning |
| Update warnings in Colab | **Active.** Colab is the primary mode, so it is exactly where a pin ages unnoticed. `local_repo_status` stays off there: the clone is fresh by construction |
| Model run state | **`fl_*` attributes injected on the model instance**, not a wrapper class: `backtest.py` calls `model.forecast(...)` directly, so a wrapper would have been far more invasive. `TimesFM_2p5` is a plain class (no `__slots__`, not a dataclass); the `fl_` prefix guards against collisions |
| `fl_recompile` | Rebuilds `ForecastConfig` from the **original** kwargs, never `dataclasses.replace` on the live config: after `compile()` that one has `max_horizon = 128`. `global_batch_size` is only set inside `compile()`, so every batch-size change must go through here |
| How the smoke test is recognised | **`count_time=False`.** It is the only caller that passes it, and the two requirements (exclude the time, do not raise `fl_degraded_after_inference`) coincide exactly with that case |
| "Run info" and the audit CSVs | Composed **in `export.py`** (`build_run_info`, `run_info_to_frame`, `save_audit_csvs`), not in the notebook cell: the cell stays pure orchestration and the fields become testable |
| Numeric acceptance criteria | **Gates on refactor identity (G1), structural identity (G2), aggregate impact with and without sign (G3/G4) and explainability (G5).** Three earlier attempts to gate on *counts of flipped scaling factors* proved uncalibratable: a flip count does not measure quality — both runs pick a point on a plateau, and which one is "right" is undecidable |
| Aggregate Motul KPI | **Sanity check, not a gate**: `BestAccuracy` is self-selected (it is the value at the `q` that same run chose), so the expected deviation is ~1e-3 pp, two to three orders of magnitude below any sensible threshold |
| Comparison baseline | **New code at batch 1**, not `main`: symmetric artefacts and batch size as the only variable. `main` stays covered by the refactor-identity gate |
| `q` on `BestAccuracyRaw == 0` SKUs | **Out of scope, but recorded**: they get `q = 0.10` from `QUANTILE_GRID` iteration order, and they are **not** confined to class C (it follows XYZ, not ABC). Worth a dedicated cycle |
| Batch-1 reference timing | **5 m 09 s** on 571 SKUs (RTX 2070 SUPER): backtest 190.9 s, forecast 94.4 s, both ~98% inference — the Python grid search costs 3.5 s, so it is not the bottleneck it was once feared to be |

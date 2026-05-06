"""
Modulo G — Backtest e ottimizzazione dello scaling factor (rolling-origin).

Il backtest simula il forecast sul passato per trovare lo scaling factor q
ottimale per ogni SKU che massimizza l'accuratezza Motul (KPI di business).

Pipeline per ogni origine di backtest:
  1. Calibrazione locale Theil-Sen (replica Modulo D, ma su storico troncato
     -> niente data leakage)
  2. Pre-indicizzazione dei dati per accesso rapido
  3. Preparazione per ogni SKU (allineamento date, calibrazione, pack size)
  4. Forecast batch con fallback automatico a chiamate per-SKU

Cross-origin:
  5. Grid search coarse (step 0.05) + fine (step 0.01 attorno al miglior q),
     accuracy mediata su tutte le origini
  6. Shrinkage opzionale: blend di q per-SKU verso la mediana globale,
     pesato dalla lunghezza dello storico (full trust a >= 36 mesi)
"""

import math

import numpy as np
import pandas as pd

from .calibration import calculate_seasonality_local, theil_sen_log_trend
from .metrics import accuracy_weighted
from .rounding import round_to_pack


ORIGIN_SHIFT = 6  # mesi di distanza tra un'origine e la successiva


def run_backtest(
    model,
    sku_series,
    df_filtered,
    *,
    id_col,
    pack_size_col,
    uom_col,
    horizon_backtest,
    min_history_points,
    n_backtest_origins,
    quantile_grid,
    calibration_months,
    rounding_mode,
    round_decimals,
    shrinkage_enabled,
    verbose=False,
):
    """
    Esegue il backtest rolling-origin e restituisce DataFrame con
    [SKU, BestQuantile, BestAccuracy, TotalWeight] per ogni SKU valido.

    Parametri (tutti keyword-only, dal Modulo A del notebook):
        model:                istanza TimesFM
        sku_series:           dict {SKU: lista_valori_storici} dal preprocessing
        df_filtered:          DataFrame long con date complete (per allineamento)
        id_col, pack_size_col, uom_col: nomi colonne
        horizon_backtest:     finestra di valutazione (mesi)
        min_history_points:   storico minimo richiesto dopo troncamento
        n_backtest_origins:   numero di origini rolling (1 = singolo split)
        quantile_grid:        lista di q per la griglia coarse (step 0.05)
        calibration_months:   lista mesi target per calibrazione stagionale
        rounding_mode:        "up" / "down" / "nearest"
        round_decimals:       decimali per arrotondamento
        shrinkage_enabled:    True per blend verso mediana globale
        verbose:              log di progresso ogni 10 SKU
    """
    # --- 1-4. Preparazione e forecast per ogni origine ---
    origins_data = _prepare_origins(
        model=model,
        sku_series=sku_series,
        df_filtered=df_filtered,
        id_col=id_col,
        pack_size_col=pack_size_col,
        uom_col=uom_col,
        horizon_backtest=horizon_backtest,
        min_history_points=min_history_points,
        n_backtest_origins=n_backtest_origins,
        calibration_months=calibration_months,
    )

    # --- 5. Grid search cross-origin ---
    results_list = _grid_search_cross_origin(
        origins_data=origins_data,
        quantile_grid=quantile_grid,
        rounding_mode=rounding_mode,
        round_decimals=round_decimals,
        verbose=verbose,
    )

    # --- 6. Shrinkage opzionale ---
    if shrinkage_enabled and results_list:
        print("Applicazione shrinkage dello scaling factor...")
        q_global = _apply_shrinkage(
            results_list=results_list,
            origins_data=origins_data,
            sku_series=sku_series,
            rounding_mode=rounding_mode,
            round_decimals=round_decimals,
        )
        print(f"  q globale (mediana): {q_global:.2f}")
        print(f"  Shrinkage applicato a {len(results_list)} SKU")

    # --- 7. Costruzione DataFrame risultato ---
    return pd.DataFrame(results_list)


def empty_backtest_results():
    """Restituisce un DataFrame vuoto con lo schema dei risultati di backtest.
    Usato dal notebook quando RUN_BACKTEST = False."""
    return pd.DataFrame(
        columns=["SKU", "BestQuantile", "BestAccuracy", "TotalWeight"]
    )


# ----------------------------------------------------------------------
# Helper interni
# ----------------------------------------------------------------------

def _prepare_origins(
    *,
    model,
    sku_series,
    df_filtered,
    id_col,
    pack_size_col,
    uom_col,
    horizon_backtest,
    min_history_points,
    n_backtest_origins,
    calibration_months,
):
    """
    Per ogni origine di backtest:
      - prepara per ogni SKU storico troncato, target, date, calibrazione locale, pack
      - lancia il forecast batch (fallback a per-SKU se necessario)
    Restituisce lista di dict {SKU: prep_dict} con base_fc valido.
    """
    grouped_bt = df_filtered.sort_values("Date").groupby(id_col)
    origins_data = []

    for origin_idx in range(n_backtest_origins):
        shift = origin_idx * ORIGIN_SHIFT
        print(f"Origine {origin_idx + 1}/{n_backtest_origins} (shift = {shift} mesi)...")

        origin_prep = {}

        for sku, values in sku_series.items():
            n = len(values)

            # Serve storico sufficiente: history + backtest window + shift
            if n <= horizon_backtest + shift:
                continue

            act_end = n - shift
            act_start = act_end - horizon_backtest

            if act_start < min_history_points:
                continue

            hist_bt = values[:act_start]
            act_bt = np.array(values[act_start:act_end], dtype=float)

            # Allineamento date dalla tabella completa
            if sku not in grouped_bt.groups:
                continue

            sku_data = grouped_bt.get_group(sku)
            full_dates = sku_data["Date"].tolist()
            trim_offset = len(full_dates) - n  # zeri iniziali rimossi a monte

            hist_bt_dates = full_dates[trim_offset : trim_offset + act_start]
            act_bt_dates = full_dates[trim_offset + act_start : trim_offset + act_end]
            test_months = [d.month for d in act_bt_dates]

            # Calibrazione locale Theil-Sen (no data leakage)
            local_factors = calculate_seasonality_local(
                hist_bt, pd.Series(hist_bt_dates), calibration_months
            )

            # Pack size per arrotondamento
            row_meta = sku_data[[pack_size_col, uom_col]].iloc[0]
            pack_val = row_meta[pack_size_col]
            pack = float(pack_val) if pd.notna(pack_val) and pack_val not in ("", None) else 1.0

            origin_prep[sku] = {
                "hist_np": np.array(hist_bt, dtype=np.float32),
                "actuals": act_bt,
                "test_months": test_months,
                "local_factors": local_factors,
                "pack": pack,
            }

        print(f"  SKU preparati: {len(origin_prep)}")

        if not origin_prep:
            origins_data.append({})
            continue

        # Forecast batch per questa origine
        bt_skus = list(origin_prep.keys())
        bt_inputs = [origin_prep[s]["hist_np"] for s in bt_skus]

        try:
            all_fc, _ = model.forecast(horizon=horizon_backtest, inputs=bt_inputs)
            for i, sku in enumerate(bt_skus):
                fc = all_fc[i]
                if hasattr(fc, "cpu"):
                    fc = fc.cpu().numpy()
                origin_prep[sku]["base_fc"] = np.array(fc, dtype=float)
        except Exception as e:
            print(f"  Batch fallito ({e}), fallback a forecast singolo...")
            for sku in bt_skus:
                try:
                    fc, _ = model.forecast(
                        horizon=horizon_backtest,
                        inputs=[origin_prep[sku]["hist_np"]],
                    )
                    fc = fc[0]
                    if hasattr(fc, "cpu"):
                        fc = fc.cpu().numpy()
                    origin_prep[sku]["base_fc"] = np.array(fc, dtype=float)
                except Exception:
                    origin_prep[sku]["base_fc"] = None

        # Tieni solo SKU con forecast valido
        origin_prep = {s: p for s, p in origin_prep.items() if p.get("base_fc") is not None}
        print(f"  SKU con forecast valido: {len(origin_prep)}")

        origins_data.append(origin_prep)

    return origins_data


def _evaluate_q(base_fc, actuals, test_months, local_factors, pack, q,
                rounding_mode, round_decimals):
    """
    Applica scaling -> calibrazione -> arrotondamento e restituisce
    (accuratezza_pesata, peso_totale).
    """
    scale = q / 0.5
    fc_scaled = base_fc * scale

    fc_calibrated = []
    for i, val in enumerate(fc_scaled):
        if i >= len(test_months):
            break
        m = test_months[i]
        factor = local_factors.get(m, 1.0)
        fc_calibrated.append(val * factor)

    fc_calibrated = np.array(fc_calibrated, dtype=float)
    fc_rounded = np.array(
        [round_to_pack(v, pack, mode=rounding_mode, decimals=round_decimals)
         for v in fc_calibrated],
        dtype=float,
    )

    acc = accuracy_weighted(actuals, fc_rounded)
    total_weight = float(np.sum(actuals + fc_rounded))
    return acc, total_weight


def _grid_search_cross_origin(*, origins_data, quantile_grid,
                              rounding_mode, round_decimals, verbose):
    """
    Griglia coarse (step 0.05 sui valori in `quantile_grid`) + fine
    (step 0.01 nell'intorno +-0.04 del miglior coarse), con accuratezza
    mediata su tutte le origini.
    """
    all_bt_skus = set()
    for od in origins_data:
        all_bt_skus.update(od.keys())

    print(f"Grid search su {len(all_bt_skus)} SKU "
          f"({len(origins_data)} origini, griglia fine attiva)...")

    results_list = []

    for idx, sku in enumerate(sorted(all_bt_skus), start=1):
        # Passo 1: griglia coarse
        coarse_acc = {}
        for q in quantile_grid:
            accs = []
            for od in origins_data:
                if sku in od:
                    acc, _ = _evaluate_q(
                        od[sku]["base_fc"], od[sku]["actuals"],
                        od[sku]["test_months"], od[sku]["local_factors"],
                        od[sku]["pack"], q,
                        rounding_mode, round_decimals,
                    )
                    accs.append(acc)
            if accs:
                coarse_acc[q] = np.mean(accs)

        if not coarse_acc:
            continue

        best_q_coarse = max(coarse_acc, key=lambda x: coarse_acc[x])

        # Passo 2: griglia fine (step 0.01 attorno al miglior coarse)
        fine_candidates = [round(best_q_coarse + d, 2)
                           for d in np.arange(-0.04, 0.05, 0.01)]
        fine_candidates = [q for q in fine_candidates
                           if 0.05 <= q <= 0.95 and q not in coarse_acc]

        fine_acc = {}
        for q in fine_candidates:
            accs = []
            for od in origins_data:
                if sku in od:
                    acc, _ = _evaluate_q(
                        od[sku]["base_fc"], od[sku]["actuals"],
                        od[sku]["test_months"], od[sku]["local_factors"],
                        od[sku]["pack"], q,
                        rounding_mode, round_decimals,
                    )
                    accs.append(acc)
            if accs:
                fine_acc[q] = np.mean(accs)

        # Unione delle due griglie: miglior q globale
        all_acc = {**coarse_acc, **fine_acc}
        best_q = max(all_acc, key=lambda x: all_acc[x])
        best_accuracy = all_acc[best_q]

        # Peso totale dall'origine primaria
        if sku in origins_data[0]:
            _, total_weight = _evaluate_q(
                origins_data[0][sku]["base_fc"], origins_data[0][sku]["actuals"],
                origins_data[0][sku]["test_months"], origins_data[0][sku]["local_factors"],
                origins_data[0][sku]["pack"], best_q,
                rounding_mode, round_decimals,
            )
        else:
            total_weight = 0.0
            for od in origins_data:
                if sku in od:
                    _, total_weight = _evaluate_q(
                        od[sku]["base_fc"], od[sku]["actuals"],
                        od[sku]["test_months"], od[sku]["local_factors"],
                        od[sku]["pack"], best_q,
                        rounding_mode, round_decimals,
                    )
                    break

        results_list.append({
            "SKU": sku,
            "BestQuantile": best_q,
            "BestAccuracy": best_accuracy,
            "TotalWeight": total_weight,
        })

        if verbose and idx % 10 == 0:
            print(f"  [{idx}] {sku} -> BestQ: {best_q}, Acc: {best_accuracy:.2f}")

    return results_list


def _apply_shrinkage(*, results_list, origins_data, sku_series,
                     rounding_mode, round_decimals):
    """
    Blend del q per-SKU verso la mediana globale, pesato dalla lunghezza
    dello storico (alpha = min(1, len/36)). Aggiorna `results_list` in-place.
    Restituisce il q globale (mediana).
    """
    q_values = [r["BestQuantile"] for r in results_list]
    q_global = float(np.median(q_values))

    for r in results_list:
        sku = r["SKU"]
        hist_len = len(sku_series.get(sku, []))
        # alpha cresce con lo storico: piena fiducia a 36 mesi
        alpha = min(1.0, hist_len / 36.0)
        q_orig = r["BestQuantile"]
        q_shrunk = round(alpha * q_orig + (1 - alpha) * q_global, 2)
        q_shrunk = max(0.05, min(0.95, q_shrunk))
        r["BestQuantile"] = q_shrunk

        # Ricalcola accuratezza al q shrinkato
        accs = []
        for od in origins_data:
            if sku in od:
                acc, _ = _evaluate_q(
                    od[sku]["base_fc"], od[sku]["actuals"],
                    od[sku]["test_months"], od[sku]["local_factors"],
                    od[sku]["pack"], q_shrunk,
                    rounding_mode, round_decimals,
                )
                accs.append(acc)
        if accs:
            r["BestAccuracy"] = np.mean(accs)

        if sku in origins_data[0]:
            _, tw = _evaluate_q(
                origins_data[0][sku]["base_fc"], origins_data[0][sku]["actuals"],
                origins_data[0][sku]["test_months"], origins_data[0][sku]["local_factors"],
                origins_data[0][sku]["pack"], q_shrunk,
                rounding_mode, round_decimals,
            )
            r["TotalWeight"] = tw

    return q_global

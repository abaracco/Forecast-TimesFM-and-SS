"""
Modulo D — Calibrazione stagionale.

La calibrazione e' bidirezionale (puo' aumentare o diminuire il forecast):
sui mesi target (es. agosto, dicembre) si stima il "residuo log-detrendizzato"
mediano e lo si applica come fattore moltiplicativo (exp(mediana_residui)).

`theil_sen_log_trend` e' l'unica implementazione canonica del trend Theil-Sen
nel progetto: viene riusata identica nel Modulo G (backtest) per garantire
coerenza con la pipeline di produzione.
"""

import math
import re

import numpy as np
import pandas as pd


def theil_sen_log_trend(values):
    """
    Stima robusta del trend su scala logaritmica (Theil-Sen completo).

    Calcola pendenza (beta) e intercetta (alpha) usando TUTTE le coppie
    di punti (vettorizzato con numpy). Filtra internamente gli zeri
    preservando le posizioni temporali originali, cosi' i residui
    restano coerenti con la serie completa.

    Restituisce (alpha, beta) oppure (None, None) se ci sono meno di 6
    valori positivi.
    """
    y = np.array(values, dtype=float)
    valid_idx = np.where(y > 0)[0]      # indici dei valori positivi
    if len(valid_idx) < 6:
        return None, None

    y_log = np.log(y[valid_idx])         # logaritmo dei valori positivi
    t = valid_idx.astype(float)          # posizioni temporali originali

    n = len(y_log)

    # Matrice delle pendenze (triangolo superiore: j > i)
    dy = y_log[None, :] - y_log[:, None]   # dy[i,j] = y_log[j] - y_log[i]
    dt = t[None, :] - t[:, None]           # dt[i,j] = t[j] - t[i]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    slopes = dy[mask] / dt[mask]           # pendenze di tutte le coppie

    if len(slopes) == 0:
        return None, None

    # Mediana delle pendenze (robusto rispetto agli outlier)
    beta = float(np.median(slopes))
    # Intercetta: mediana dei residui (y_log - beta * t)
    alpha = float(np.median(y_log - beta * t))
    return alpha, beta


def month_from_label(label):
    """Estrae il numero del mese (1-12) da un'etichetta 'YYYY_MM'."""
    try:
        return int(str(label).split("_")[1])
    except Exception:
        return None


def compute_calibration_factors(df_filtered, id_col, calibration_months):
    """
    Calcola i fattori di calibrazione stagionale, sia globali che per-SKU.

    Pipeline:
      1. Pivot dello storico in formato wide (una colonna per mese 'YYYY_MM')
      2. Per ogni SKU con storico sufficiente (>=6 valori positivi):
         - Stima trend Theil-Sen log-lineare
         - Per ogni mese target, calcola residui = log(reale) - trend
      3. Aggrega:
         - Globale: mediana dei residui di TUTTI gli SKU per mese target
         - Per-SKU: mediana dei residui dello specifico SKU (se ha >=2 osservazioni
           per quel mese), altrimenti fallback al fattore globale
      4. Converte residui in fattori moltiplicativi: factor = exp(mediana_residuo)

    Parametri:
        df_filtered:           DataFrame in formato long con colonne ID_COL, Period, Demand
        id_col:                nome della colonna SKU (es. "SKU")
        calibration_months:    lista di interi 1-12 dei mesi da calibrare; [] disabilita

    Restituisce:
        sku_factors:    dict {sku: {mese: fattore}}
        global_factors: dict {mese: fattore}
    """
    if not calibration_months:
        return {}, {}

    # --- 1. Pivot in formato wide ---
    df_hist_wide = df_filtered.pivot(
        index=id_col,
        columns="Period",
        values="Demand"
    ).reset_index()

    time_cols = [c for c in df_hist_wide.columns if re.match(r"^\d{4}_\d{2}$", str(c))]
    time_cols = sorted(time_cols)

    # --- 2. Fattore globale: mediana dei residui log per mese target ---
    global_med_residuals = _global_month_medians(
        df_hist_wide, time_cols, calibration_months
    )
    global_factors = {
        m: math.exp(global_med_residuals[m]) for m in calibration_months
    }

    # --- 3. Fattori per-SKU ---
    sku_factors = {}
    hist_by_sku = df_hist_wide.set_index(id_col)

    for sku, row_hist in hist_by_sku.iterrows():
        vals = pd.Series([row_hist[c] for c in time_cols], index=time_cols, dtype="float")
        vals = pd.to_numeric(vals, errors="coerce")

        nz = np.where((vals.fillna(0) > 0).values)[0]
        if len(nz) == 0:
            continue

        # Trim solo zeri iniziali, mantieni zeri interni
        vals_trimmed = vals.iloc[nz[0]:].dropna()

        if (vals_trimmed > 0).sum() >= 6:
            alpha, beta = theil_sen_log_trend(vals_trimmed.values)
            if alpha is not None:
                bucket = {m: [] for m in calibration_months}
                for i, (c, v) in enumerate(vals_trimmed.items()):
                    m = month_from_label(c)
                    if m in bucket and v > 0:
                        r = np.log(v) - (alpha + beta * i)
                        bucket[m].append(r)

                per_month_med = {}
                for m in calibration_months:
                    if len(bucket[m]) >= 2:
                        # Almeno 2 osservazioni -> fattore specifico per SKU
                        per_month_med[m] = float(np.median(bucket[m]))
                    else:
                        # Meno di 2 osservazioni -> fallback al fattore globale
                        per_month_med[m] = global_med_residuals[m]
            else:
                per_month_med = {m: global_med_residuals[m] for m in calibration_months}
        else:
            per_month_med = {m: global_med_residuals[m] for m in calibration_months}

        # Converte residui log in fattori moltiplicativi
        sku_factors[sku] = {
            m: math.exp(per_month_med[m]) for m in calibration_months
        }

    return sku_factors, global_factors


def _global_month_medians(df_wide, time_cols, target_months):
    """
    Calcola la mediana dei residui log-detrendizzati per ogni mese target,
    aggregando tutti gli SKU con storico sufficiente.
    Restituisce dict {mese: mediana_residui} (zero se nessun dato).
    """
    med = {m: [] for m in target_months}

    for _, row in df_wide.iterrows():
        vals = pd.Series([row[c] for c in time_cols], index=time_cols, dtype="float")
        vals = pd.to_numeric(vals, errors="coerce")

        # Trim solo zeri iniziali, mantieni zeri interni
        nz = np.where((vals.fillna(0) > 0).values)[0]
        if len(nz) == 0:
            continue
        vals = vals.iloc[nz[0]:].dropna()

        if (vals > 0).sum() < 6:
            continue

        alpha, beta = theil_sen_log_trend(vals.values)
        if alpha is None:
            continue

        # Calcola residuo = log(reale) - trend stimato, per ogni mese target
        for i, (c, v) in enumerate(vals.items()):
            m = month_from_label(c)
            if m in med and v > 0:
                r = np.log(v) - (alpha + beta * i)
                med[m].append(r)

    return {
        m: (float(np.median(v)) if v else 0.0)
        for m, v in med.items()
    }


def get_calibration_factor(sku, month, sku_factors, global_factors, calibration_months):
    """
    Restituisce il fattore di calibrazione per uno SKU e un mese (1-12).
    Logica di priorita':
      1. Fattore specifico per-SKU (se disponibile)
      2. Fattore globale del mese (fallback)
      3. 1.0 (nessuna calibrazione)
    """
    if not calibration_months:
        return 1.0

    # Se il mese non rientra tra quelli da calibrare -> nessun aggiustamento
    if month not in calibration_months:
        return 1.0

    # Priorita' 1: fattore per-SKU
    if sku in sku_factors and month in sku_factors[sku]:
        return sku_factors[sku][month]

    # Priorita' 2: fattore globale
    if month in global_factors:
        return global_factors[month]

    return 1.0


def calculate_seasonality_local(hist_bt, dates_bt, calibration_months):
    """
    Calcola fattori di calibrazione stagionale usando SOLO uno storico
    troncato (versione "no data leakage" per il backtest del Modulo G).

    Replica la logica di `compute_calibration_factors` ma per un singolo
    SKU su dati limitati. Usa `theil_sen_log_trend` per coerenza con
    la pipeline di produzione.

    Parametri:
        hist_bt:             lista/array di valori storici troncati
        dates_bt:            pd.Series o lista di date corrispondenti
        calibration_months:  lista di mesi target (1-12); [] disabilita

    Restituisce dict {mese: fattore_moltiplicativo}.
    """
    if not calibration_months:
        return {}

    ts = pd.Series(hist_bt, index=dates_bt)
    ts = ts[ts > 0]

    if len(ts) < 6:
        return {}

    alpha, beta = theil_sen_log_trend(hist_bt)

    if alpha is None:
        return {}

    month_residuals = {m: [] for m in calibration_months}

    dates_list = list(dates_bt)
    for date, val in ts.items():
        m = date.month
        if m in calibration_months:
            try:
                t = dates_list.index(date)
                expected_log = alpha + beta * t
                actual_log = np.log(val)
                res = actual_log - expected_log
                month_residuals[m].append(res)
            except ValueError:
                continue

    factors = {}
    for m in calibration_months:
        residuals = month_residuals[m]
        if len(residuals) >= 1:
            med_res = np.median(residuals)
            factors[m] = math.exp(med_res)
        else:
            factors[m] = 1.0

    return factors

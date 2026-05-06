"""
Modulo B — Caricamento e preprocessing.

Trasforma il file Excel (formato wide: una colonna per mese 'YYYY_MM') nel
DataFrame long che alimenta tutto il resto della pipeline, applicando:
  1. Identificazione automatica delle colonne temporali
  2. NaN -> 0 sulle colonne temporali
  3. Wide -> Long
  4. Parsing delle date e ordinamento
  5. Filtro SKU con storico insufficiente
  6. Winsorizing degli outlier per SKU (se abilitato)

`build_sku_series` chiude la fase preparando il dizionario {SKU: lista_valori}
nel formato richiesto da TimesFM (con trim opzionale degli zeri iniziali).
"""

import re

import numpy as np
import pandas as pd


DATE_PATTERN = re.compile(r"^\d{4}_\d{2}$")


def detect_date_columns(df):
    """
    Restituisce (date_cols, meta_cols) identificando le colonne temporali
    secondo il pattern 'YYYY_MM'.
    """
    date_cols = [col for col in df.columns if DATE_PATTERN.match(str(col))]
    meta_cols = [col for col in df.columns if col not in date_cols]
    return date_cols, meta_cols


def winsorize_series(s, level, enabled=True):
    """
    Applica winsorizing a una serie: taglia i valori sotto il percentile
    `level` e sopra il percentile `1-level`.
    Se `enabled` e' False, restituisce la serie invariata.
    """
    if not enabled:
        return s

    vals = s.dropna()
    if len(vals) == 0:
        return s

    q_low = vals.quantile(level)
    q_high = vals.quantile(1 - level)

    return s.clip(lower=q_low, upper=q_high)


def wide_to_long(df_raw, id_col):
    """
    Pipeline B.2 -> B.5: identifica colonne temporali, mette NaN -> 0,
    converte da wide a long, fa parsing delle date e ordina cronologicamente.

    Restituisce (df_long, date_cols, meta_cols).
    """
    date_cols, meta_cols = detect_date_columns(df_raw)

    # B.3 — NaN -> 0 sulle colonne temporali
    df_raw = df_raw.copy()
    df_raw[date_cols] = df_raw[date_cols].fillna(0)

    # B.4 — wide -> long
    df_long = df_raw.melt(
        id_vars=meta_cols,
        value_vars=date_cols,
        var_name="Period",
        value_name="Demand"
    )
    df_long = df_long.dropna(subset=["Demand"]).reset_index(drop=True)

    # B.5 — parsing date e ordinamento
    df_long["Date"] = pd.to_datetime(df_long["Period"], format="%Y_%m")
    df_long = df_long.sort_values(by=[id_col, "Date"]).reset_index(drop=True)

    return df_long, date_cols, meta_cols


def filter_min_history(df_long, id_col, min_history_points):
    """
    Filtra gli SKU con storico inferiore a `min_history_points`.
    Restituisce (df_filtered, n_total, n_kept).
    """
    hist_counts = (
        df_long.groupby(id_col)["Demand"]
        .count()
        .reset_index(name="history_points")
    )
    valid_skus = hist_counts[hist_counts["history_points"] >= min_history_points][id_col]

    n_total = df_long[id_col].nunique()
    n_kept = len(valid_skus)

    df_filtered = (
        df_long[df_long[id_col].isin(valid_skus)]
        .reset_index(drop=True)
        .copy()
    )

    return df_filtered, n_total, n_kept


def apply_winsorize(df_filtered, id_col, level, enabled):
    """
    Applica winsorize_series per SKU in-place sulla colonna 'Demand'.
    Se `enabled` e' False non modifica nulla.
    Restituisce il DataFrame modificato.
    """
    df_filtered = df_filtered.copy()
    df_filtered["Demand"] = (
        df_filtered
        .groupby(id_col)["Demand"]
        .transform(lambda s: winsorize_series(s, level, enabled=enabled))
    )
    return df_filtered


def build_sku_series(df_filtered, id_col, trim_leading_zeros):
    """
    Costruisce il dizionario {SKU: lista_valori_storici} nel formato
    atteso da TimesFM. Se `trim_leading_zeros` e' True, rimuove gli zeri
    in testa alla serie (periodo pre-lancio prodotto). Zeri interni e
    finali sono SEMPRE mantenuti.
    """
    sku_series = {}

    for sku, group in df_filtered.groupby(id_col):
        g = group.sort_values("Date")
        values = g["Demand"].astype(float).tolist()

        if trim_leading_zeros:
            idx = 0
            while idx < len(values) and values[idx] == 0:
                idx += 1
            values = values[idx:]

        if len(values) == 0:
            continue

        sku_series[sku] = values

    return sku_series


def build_backtest_series(sku_series, horizon_backtest, min_history_points):
    """
    Prepara serie storiche TRONCATE e relativi target per il backtest.

    Per ogni SKU divide la serie in:
      - hist_bt:  tutti i valori tranne gli ultimi `horizon_backtest` mesi
                  (input al modello durante il backtest)
      - act_bt:   gli ultimi `horizon_backtest` mesi
                  (target per il calcolo dell'accuratezza)

    Esclude gli SKU che, dopo il troncamento, non avrebbero almeno
    `min_history_points` mesi nello storico.

    Restituisce:
        backtest_series:  dict {SKU: lista_valori_troncati}
        backtest_actuals: dict {SKU: np.array dei valori reali target}
        skipped:          numero di SKU scartati per serie troppo corta
    """
    backtest_series = {}
    backtest_actuals = {}
    skipped = 0

    for sku, values in sku_series.items():
        n = len(values)

        if n <= horizon_backtest:
            skipped += 1
            continue

        hist_bt = values[:-horizon_backtest]
        act_bt = values[-horizon_backtest:]

        if len(hist_bt) < min_history_points:
            skipped += 1
            continue

        backtest_series[sku] = hist_bt
        backtest_actuals[sku] = np.array(act_bt, dtype=float)

    return backtest_series, backtest_actuals, skipped

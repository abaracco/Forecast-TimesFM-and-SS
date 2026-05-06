"""
Modulo J — Costruzione tabella finale ed export Excel.

La parte di I/O specifica per ambiente (download Colab vs save dialog
locale) resta nel notebook. Qui vive solo la logica di:
  1. Pivot dello storico in formato wide
  2. Rinomina colonne forecast (prefisso 'f' per distinguerle dallo storico)
  3. Merge: metadati + inventario + storico + forecast
  4. Salvataggio Excel
"""

import re

import pandas as pd


def build_forecast_wide(df_fc_long, id_col):
    """
    Converte il forecast da long a wide ordinando le colonne mese
    cronologicamente.

    Parametri:
        df_fc_long: DataFrame con colonne [id_col, "Period", "Forecast"]
        id_col:     nome colonna SKU
    """
    df_fc_wide = df_fc_long.pivot(
        index=id_col,
        columns="Period",
        values="Forecast"
    ).reset_index()

    forecast_cols = sorted(
        [c for c in df_fc_wide.columns if re.match(r"^\d{4}_\d{2}$", str(c))]
    )
    return df_fc_wide[[id_col] + forecast_cols]


def build_history_wide(df_filtered, id_col):
    """
    Converte lo storico (long) in formato wide pivotando su 'Period'.
    """
    return df_filtered.pivot(
        index=id_col,
        columns="Period",
        values="Demand"
    ).reset_index()


def build_final_table(
    df_filtered,
    df_fc_wide,
    df_inventory,
    *,
    id_col,
    desc_col,
    pack_size_col,
    uom_col,
):
    """
    Merge finale: metadati base + inventario (opzionale) + storico + forecast.

    Le colonne forecast vengono prefissate con 'f' per distinguerle dalle
    colonne storiche con lo stesso pattern 'YYYY_MM'.

    Parametri:
        df_filtered:    storico in formato long (post winsorize)
        df_fc_wide:     forecast in formato wide
        df_inventory:   risultato di calculate_inventory_logic, oppure None se disabilitato
        id_col, desc_col, pack_size_col, uom_col: nomi colonne (dal Modulo A)
    """
    # 1. Metadati base
    meta = df_filtered[[id_col, desc_col, pack_size_col, uom_col]].drop_duplicates()

    # 2. Inventario (se calcolato)
    if df_inventory is not None:
        df_inv_clean = df_inventory.rename(columns={"LT_Final": "LT"})
        out_step1 = meta.merge(df_inv_clean, on=id_col, how="left")
    else:
        out_step1 = meta.copy()

    # 3. Storico in formato wide
    df_hist_wide = build_history_wide(df_filtered, id_col)

    # 4. Forecast in formato wide con prefisso 'f'
    df_fc_pref = df_fc_wide.copy()
    df_fc_pref.rename(
        columns={c: f"f{c}" for c in df_fc_pref.columns if c != id_col},
        inplace=True
    )

    # 5. Merge finale
    out_final = out_step1.merge(df_hist_wide, on=id_col, how="left")
    out_final = out_final.merge(df_fc_pref, on=id_col, how="left")

    # SKU come stringa per evitare problemi di formattazione in Excel
    out_final[id_col] = out_final[id_col].astype(str)

    # NaN nella safety stock -> 0
    if "SafetyStock" in out_final.columns:
        out_final["SafetyStock"] = out_final["SafetyStock"].fillna(0)

    return out_final


def save_excel(df, path):
    """Salva il DataFrame in formato Excel (un solo foglio, indice escluso)."""
    df.to_excel(path, index=False)

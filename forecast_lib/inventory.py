"""
Modulo I — Pianificazione inventario (ABC, XYZ, scorta di sicurezza).

Pipeline:
  1. Preparazione dati (ultimi `ss_lookback_months` mesi di storico)
  2. Classificazione ABC (Pareto sui volumi cumulativi)
     Guard: se il volume totale del lookback e' zero -> tutti gli SKU in classe C
  3. Classificazione XYZ (coefficiente di variazione della domanda)
  4. Classe combinata ABC/XYZ -> livello di servizio target dalla matrice
  5. Calcolo scorta di sicurezza:
        SS = Z * sigma * sqrt((LT + ReorderPeriod) / 30)
     arrotondata SEMPRE per eccesso al multiplo d'imballo, indipendentemente
     dal `rounding_mode` impostato per il forecast.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .rounding import round_to_pack


def calculate_inventory_logic(
    df_history,
    meta_df,
    *,
    id_col,
    lt_col_name,
    pack_size_col,
    abc_limits,
    xyz_limits,
    service_level_matrix,
    ss_lookback_months,
    default_lead_time,
    reorder_period,
    round_decimals,
):
    """
    Calcola classificazione ABC/XYZ e scorta di sicurezza per ogni SKU.

    Parametri (tutti keyword-only, presi dal Modulo A del notebook):
        df_history:           DataFrame long con colonne [id_col, "Date", "Demand"]
        meta_df:              DataFrame metadati (deve contenere id_col;
                              opzionalmente lt_col_name e pack_size_col)
        id_col:               nome colonna SKU (es. "SKU")
        lt_col_name:          nome colonna lead time (es. "LT")
        pack_size_col:        nome colonna pack size (es. "Round")
        abc_limits:           dict {"A": 0.70, "B": 0.90, "C": 1.00}
        xyz_limits:           dict {"X": 0.40, "Y": 0.80, "Z": 999.0}
        service_level_matrix: dict {"AX": 0.97, ..., "CZ": 0.0}
        ss_lookback_months:   mesi di storico usati per media e std
        default_lead_time:    LT di fallback (giorni) se la colonna manca
        reorder_period:       periodo di riordino (giorni)
        round_decimals:       decimali per l'arrotondamento finale

    Restituisce DataFrame con colonne [id_col, "LT_Final", "ABC", "XYZ", "SafetyStock"].
    """
    # --- 1. Preparazione dati (ultimi N mesi di storico winsorizzato) ---
    last_date = df_history["Date"].max()
    start_date_ss = last_date - pd.DateOffset(months=ss_lookback_months - 1)

    df_ss = df_history[df_history["Date"] >= start_date_ss].copy()

    # Statistiche per SKU
    stats = df_ss.groupby(id_col)["Demand"].agg(["sum", "mean", "std"]).reset_index()
    stats.rename(columns={"sum": "TotalVol", "mean": "AvgDemand", "std": "StdDev"},
                 inplace=True)
    stats["StdDev"] = stats["StdDev"].fillna(0)

    # --- 2. Classificazione ABC (Pareto sui volumi) ---
    stats = stats.sort_values("TotalVol", ascending=False)
    total_volume_global = stats["TotalVol"].sum()

    if total_volume_global <= 0:
        # Guardia: se nessun volume nel lookback, tutti gli SKU -> classe C
        stats["ABC"] = "C"
    else:
        stats["CumVol"] = stats["TotalVol"].cumsum()
        stats["CumPerc"] = stats["CumVol"] / total_volume_global

        def get_abc(p):
            if p <= abc_limits["A"]:
                return "A"
            elif p <= abc_limits["B"]:
                return "B"
            return "C"

        stats["ABC"] = stats["CumPerc"].apply(get_abc)

    # --- 3. Classificazione XYZ (volatilita') ---
    stats["CV"] = stats["StdDev"] / stats["AvgDemand"]
    stats["CV"] = stats["CV"].replace([np.inf, -np.inf], 999.0).fillna(0)

    def get_xyz(cv):
        if cv <= xyz_limits["X"]:
            return "X"
        elif cv <= xyz_limits["Y"]:
            return "Y"
        return "Z"

    stats["XYZ"] = stats["CV"].apply(get_xyz)

    # --- 4. Classe combinata ---
    stats["Class"] = stats["ABC"] + stats["XYZ"]

    # --- 5. Recupero lead time e pack size ---
    if lt_col_name in meta_df.columns:
        lt_map = meta_df[[id_col, lt_col_name]].drop_duplicates(subset=id_col)
        stats = stats.merge(lt_map, on=id_col, how="left")
        stats["LT_Final"] = stats[lt_col_name].fillna(default_lead_time)
    else:
        stats["LT_Final"] = default_lead_time

    if pack_size_col in meta_df.columns:
        pack_map = meta_df[[id_col, pack_size_col]].drop_duplicates(subset=id_col)
        stats = stats.merge(pack_map, on=id_col, how="left")
    else:
        stats[pack_size_col] = 1.0

    # --- 6. Calcolo scorta di sicurezza ---
    # Formula: SS = Z * sigma * sqrt((LT + ReorderPeriod) / 30)
    # Arrotondata PER ECCESSO al multiplo d'imballo (sempre "up")
    ss_values = []

    for _, row in stats.iterrows():
        cls = row["Class"]
        sigma = row["StdDev"]
        lt = row["LT_Final"]

        pack_val = row.get(pack_size_col, 1.0)
        try:
            pack = float(pack_val) if pd.notna(pack_val) and pack_val > 0 else 1.0
        except Exception:
            pack = 1.0

        sl_target = service_level_matrix.get(cls, 0.0)

        if sl_target <= 0 or sigma <= 0:
            ss_values.append(0.0)
            continue

        z_score = norm.ppf(sl_target)
        time_factor = np.sqrt((lt + reorder_period) / 30.0)

        ss_raw = z_score * sigma * time_factor
        # Arrotondamento SEMPRE per eccesso, a prescindere dal ROUNDING_MODE
        ss_final = round_to_pack(ss_raw, pack, mode="up", decimals=round_decimals)

        ss_values.append(ss_final)

    stats["SafetyStock"] = ss_values

    cols_to_keep = [id_col, "LT_Final", "ABC", "XYZ", "SafetyStock"]
    return stats[cols_to_keep]

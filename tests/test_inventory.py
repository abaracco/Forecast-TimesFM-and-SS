"""
Test su calculate_inventory_logic.

Coperti:
  - mini-dataset con classi attese note (A, C)
  - guard ABC: tutti gli SKU -> C se volume globale e' zero
  - SS = 0 per classe CZ (livello servizio 0%)
  - SS arrotondata SEMPRE per eccesso al pack size
  - LT preso dalla colonna se presente, altrimenti default
"""

import numpy as np
import pandas as pd

from forecast_lib.inventory import calculate_inventory_logic


# Configurazioni standard usate dai test (replicano il Modulo A del notebook)
ABC_LIMITS = {"A": 0.70, "B": 0.90, "C": 1.00}
XYZ_LIMITS = {"X": 0.40, "Y": 0.80, "Z": 999.0}
SERVICE_LEVEL_MATRIX = {
    "AX": 0.97, "AY": 0.95, "AZ": 0.93,
    "BX": 0.91, "BY": 0.90, "BZ": 0.89,
    "CX": 0.87, "CY": 0.80, "CZ": 0.00,
}


def _make_history(skus_demands):
    """Costruisce un DataFrame storico long da {sku: lista_valori_mensili}."""
    rows = []
    base_date = pd.Timestamp("2023-01-01")
    for sku, demands in skus_demands.items():
        for i, d in enumerate(demands):
            rows.append({
                "SKU": sku,
                "Date": base_date + pd.DateOffset(months=i),
                "Demand": float(d),
            })
    return pd.DataFrame(rows)


def _make_meta(skus, lt=None, pack=None):
    """Costruisce un DataFrame metadati con LT e Round opzionali."""
    df = pd.DataFrame({"SKU": skus})
    if lt is not None:
        df["LT"] = [lt] * len(skus) if not isinstance(lt, list) else lt
    if pack is not None:
        df["Round"] = [pack] * len(skus) if not isinstance(pack, list) else pack
    return df


def _common_kwargs():
    return dict(
        id_col="SKU",
        lt_col_name="LT",
        pack_size_col="Round",
        abc_limits=ABC_LIMITS,
        xyz_limits=XYZ_LIMITS,
        service_level_matrix=SERVICE_LEVEL_MATRIX,
        ss_lookback_months=12,
        default_lead_time=30,
        reorder_period=30,
        round_decimals=3,
    )


# ----------------------------------------------------------------------
# Classificazione
# ----------------------------------------------------------------------

def test_volume_zero_global_falls_to_class_c():
    # Tutti gli SKU con domanda zero -> guard: classe C
    df_history = _make_history({
        "S1": [0] * 12,
        "S2": [0] * 12,
    })
    meta = _make_meta(["S1", "S2"], lt=30, pack=1)
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())
    assert (out["ABC"] == "C").all()


def test_pareto_abc_classification_is_correct():
    # 3 SKU con volumi tali da finire chiaramente in A, B, C:
    #   S_A: 72 (cumPerc 0.60 -> A)
    #   S_B: 30 (cumPerc 0.85 -> B)
    #   S_C: 18 (cumPerc 1.00 -> C)
    df_history = _make_history({
        "S_A": [6] * 12,    # totale 72
        "S_B": [2.5] * 12,  # totale 30
        "S_C": [1.5] * 12,  # totale 18
    })
    meta = _make_meta(["S_A", "S_B", "S_C"], lt=30, pack=1)
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())

    abc_map = dict(zip(out["SKU"], out["ABC"]))
    assert abc_map["S_A"] == "A"
    assert abc_map["S_B"] == "B"
    assert abc_map["S_C"] == "C"

    # Domanda perfettamente costante -> CV = 0 -> X
    xyz_map = dict(zip(out["SKU"], out["XYZ"]))
    assert xyz_map["S_A"] == "X"


def test_erratic_demand_is_z():
    # CV alto -> Z
    df_history = _make_history({
        "S1": [100, 0, 200, 0, 300, 0, 400, 0, 500, 0, 600, 0],
    })
    meta = _make_meta(["S1"], lt=30, pack=1)
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())
    assert out["XYZ"].iloc[0] == "Z"


# ----------------------------------------------------------------------
# Safety stock
# ----------------------------------------------------------------------

def test_cz_class_has_zero_safety_stock():
    # Classe CZ -> service level 0 -> SS = 0
    # (volume basso + alta variabilita')
    df_history = _make_history({
        "DOM": [1000] * 12,                                   # spinge DOM in A
        "S1":  [10, 0, 50, 0, 5, 100, 0, 30, 0, 80, 0, 10],   # bassa quota, alta CV
    })
    meta = _make_meta(["DOM", "S1"], lt=30, pack=1)
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())
    s1_row = out[out["SKU"] == "S1"].iloc[0]
    assert s1_row["ABC"] == "C"
    assert s1_row["XYZ"] == "Z"
    assert s1_row["SafetyStock"] == 0.0


def test_safety_stock_rounded_up_to_pack():
    # Anche se rounding_mode di sistema fosse "down", SS e' SEMPRE "up"
    # (qui non passiamo rounding_mode, e' implicito "up" dentro inventory)
    df_history = _make_history({
        "S1": [100, 110, 105, 95, 100, 102, 98, 103, 100, 99, 101, 100],
    })
    meta = _make_meta(["S1"], lt=30, pack=10)  # pack 10
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())
    ss = out.iloc[0]["SafetyStock"]
    # SS deve essere multiplo di 10
    assert ss % 10 == 0
    # E deve essere > 0 (la classe e' AX -> SL=97%, sigma>0)
    assert ss > 0


def test_lt_from_column_overrides_default():
    df_history = _make_history({
        "S1": [100, 110, 105, 95, 100, 102, 98, 103, 100, 99, 101, 100],
    })
    meta_with_lt = _make_meta(["S1"], lt=60, pack=1)
    meta_no_lt = pd.DataFrame({"SKU": ["S1"], "Round": [1]})

    kwargs = _common_kwargs()
    out_with = calculate_inventory_logic(df_history, meta_with_lt, **kwargs)
    out_without = calculate_inventory_logic(df_history, meta_no_lt, **kwargs)

    # Con LT=60 dovrebbe esserci piu' safety stock che con LT=30 (default)
    assert out_with.iloc[0]["LT_Final"] == 60
    assert out_without.iloc[0]["LT_Final"] == 30


def test_zero_sigma_gives_zero_safety_stock():
    # Domanda perfettamente costante -> sigma=0 -> SS=0
    df_history = _make_history({
        "S1": [100] * 12,
    })
    meta = _make_meta(["S1"], lt=30, pack=1)
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())
    assert out.iloc[0]["SafetyStock"] == 0.0


def test_output_columns():
    df_history = _make_history({"S1": [100] * 12})
    meta = _make_meta(["S1"], lt=30, pack=1)
    out = calculate_inventory_logic(df_history, meta, **_common_kwargs())
    expected_cols = {"SKU", "LT_Final", "ABC", "XYZ", "SafetyStock"}
    assert set(out.columns) == expected_cols

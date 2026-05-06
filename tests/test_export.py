"""
Test su build_final_table: merge corretto di metadati, inventario,
storico e forecast con prefisso 'f' sulle colonne previste.
"""

import pandas as pd

from forecast_lib.export import build_final_table, build_forecast_wide


def test_build_forecast_wide_orders_columns():
    df_long = pd.DataFrame({
        "SKU": ["A", "A", "A"],
        "Period": ["2025_03", "2025_01", "2025_02"],
        "Forecast": [30, 10, 20],
    })
    out = build_forecast_wide(df_long, id_col="SKU")
    # le colonne forecast devono essere ordinate cronologicamente
    cols = [c for c in out.columns if c != "SKU"]
    assert cols == ["2025_01", "2025_02", "2025_03"]


def test_build_final_table_basic_merge():
    # storico
    df_filtered = pd.DataFrame({
        "SKU": ["A", "A", "B", "B"],
        "Description": ["aaa", "aaa", "bbb", "bbb"],
        "Round": [6, 6, 12, 12],
        "BUn": ["EA", "EA", "EA", "EA"],
        "Period": ["2024_01", "2024_02"] * 2,
        "Demand": [10, 15, 20, 25],
        "Date": pd.to_datetime(["2024_01", "2024_02"] * 2, format="%Y_%m"),
    })

    # forecast wide
    df_fc_wide = pd.DataFrame({
        "SKU": ["A", "B"],
        "2025_01": [12.0, 22.0],
        "2025_02": [14.0, 24.0],
    })

    # inventory
    df_inventory = pd.DataFrame({
        "SKU": ["A", "B"],
        "LT_Final": [30, 45],
        "ABC": ["A", "B"],
        "XYZ": ["X", "Y"],
        "SafetyStock": [12.0, 0.0],
    })

    out = build_final_table(
        df_filtered, df_fc_wide, df_inventory,
        id_col="SKU", desc_col="Description",
        pack_size_col="Round", uom_col="BUn",
    )

    # SKU come stringa (dtype puo' essere object o StringDtype in pandas moderno)
    assert all(isinstance(v, str) for v in out["SKU"])

    # colonne forecast devono avere prefisso 'f'
    assert "f2025_01" in out.columns
    assert "f2025_02" in out.columns

    # colonne storiche senza prefisso
    assert "2024_01" in out.columns
    assert "2024_02" in out.columns

    # LT_Final rinominato in LT
    assert "LT" in out.columns

    # 2 righe (una per SKU)
    assert len(out) == 2


def test_build_final_table_without_inventory():
    df_filtered = pd.DataFrame({
        "SKU": ["A"],
        "Description": ["aaa"],
        "Round": [6],
        "BUn": ["EA"],
        "Period": ["2024_01"],
        "Demand": [10],
        "Date": pd.to_datetime(["2024_01"], format="%Y_%m"),
    })
    df_fc_wide = pd.DataFrame({"SKU": ["A"], "2025_01": [12.0]})

    out = build_final_table(
        df_filtered, df_fc_wide, df_inventory=None,
        id_col="SKU", desc_col="Description",
        pack_size_col="Round", uom_col="BUn",
    )

    # niente colonne inventario
    assert "LT" not in out.columns
    assert "ABC" not in out.columns
    assert "SafetyStock" not in out.columns
    # ma forecast e storico ci sono
    assert "f2025_01" in out.columns
    assert "2024_01" in out.columns


def test_build_final_table_fills_missing_safety_stock_with_zero():
    # SKU 'B' non e' nell'inventario -> SafetyStock NaN -> deve diventare 0
    df_filtered = pd.DataFrame({
        "SKU": ["A", "B"],
        "Description": ["aaa", "bbb"],
        "Round": [6, 12],
        "BUn": ["EA", "EA"],
        "Period": ["2024_01", "2024_01"],
        "Demand": [10, 20],
        "Date": pd.to_datetime(["2024_01", "2024_01"], format="%Y_%m"),
    })
    df_fc_wide = pd.DataFrame({"SKU": ["A", "B"], "2025_01": [12.0, 22.0]})
    df_inventory = pd.DataFrame({
        "SKU": ["A"],   # solo A
        "LT_Final": [30],
        "ABC": ["A"], "XYZ": ["X"], "SafetyStock": [10.0],
    })

    out = build_final_table(
        df_filtered, df_fc_wide, df_inventory,
        id_col="SKU", desc_col="Description",
        pack_size_col="Round", uom_col="BUn",
    )

    b_row = out[out["SKU"] == "B"].iloc[0]
    assert b_row["SafetyStock"] == 0.0

"""
Test su preprocessing: winsorize, wide-to-long, filtro storia,
costruzione sku_series con trim degli zeri iniziali.
"""

import math

import numpy as np
import pandas as pd

from forecast_lib.preprocessing import (
    apply_winsorize,
    build_backtest_series,
    build_sku_series,
    detect_date_columns,
    filter_min_history,
    wide_to_long,
    winsorize_series,
)


# ----------------------------------------------------------------------
# winsorize_series
# ----------------------------------------------------------------------

def test_winsorize_disabled_returns_unchanged():
    s = pd.Series([1, 100, 2, 3, 4])
    result = winsorize_series(s, 0.05, enabled=False)
    pd.testing.assert_series_equal(result, s)


def test_winsorize_clips_extremes():
    # serie con un outlier alto e uno basso
    s = pd.Series([1, 50, 50, 50, 50, 50, 50, 50, 50, 999])
    result = winsorize_series(s, 0.10, enabled=True)
    # Quantile 10% di [1,50,...,999] = 32.4 (interpolazione)
    # Quantile 90% = ~501.5
    # Quindi 1 viene alzato e 999 abbassato
    assert result.iloc[0] > 1
    assert result.iloc[-1] < 999


def test_winsorize_empty_series_returns_empty():
    s = pd.Series([], dtype=float)
    result = winsorize_series(s, 0.05, enabled=True)
    assert len(result) == 0


# ----------------------------------------------------------------------
# detect_date_columns
# ----------------------------------------------------------------------

def test_detects_yyyy_mm_pattern():
    df = pd.DataFrame(columns=["SKU", "Description", "2024_01", "2024_02", "LT"])
    date_cols, meta_cols = detect_date_columns(df)
    assert date_cols == ["2024_01", "2024_02"]
    assert meta_cols == ["SKU", "Description", "LT"]


def test_no_date_columns():
    df = pd.DataFrame(columns=["SKU", "Description", "LT"])
    date_cols, meta_cols = detect_date_columns(df)
    assert date_cols == []
    assert meta_cols == ["SKU", "Description", "LT"]


# ----------------------------------------------------------------------
# wide_to_long (test integrato)
# ----------------------------------------------------------------------

def test_wide_to_long_basic():
    df_raw = pd.DataFrame({
        "SKU": ["A", "B"],
        "Description": ["aaa", "bbb"],
        "2024_01": [10, 20],
        "2024_02": [15, 25],
    })
    df_long, date_cols, meta_cols = wide_to_long(df_raw, id_col="SKU")
    assert len(df_long) == 4  # 2 SKU x 2 mesi
    assert set(df_long.columns) >= {"SKU", "Description", "Period", "Demand", "Date"}
    assert df_long["Period"].tolist() == ["2024_01", "2024_02", "2024_01", "2024_02"]


def test_wide_to_long_replaces_nan_with_zero():
    df_raw = pd.DataFrame({
        "SKU": ["A"],
        "2024_01": [np.nan],
        "2024_02": [10],
    })
    df_long, _, _ = wide_to_long(df_raw, id_col="SKU")
    assert df_long.loc[df_long["Period"] == "2024_01", "Demand"].iloc[0] == 0


# ----------------------------------------------------------------------
# filter_min_history
# ----------------------------------------------------------------------

def test_filter_min_history_keeps_only_long_enough():
    df_long = pd.DataFrame({
        "SKU": ["A"] * 10 + ["B"] * 3,
        "Date": list(pd.date_range("2024-01-01", periods=10, freq="MS")) +
                list(pd.date_range("2024-01-01", periods=3, freq="MS")),
        "Demand": list(range(10)) + list(range(3)),
    })
    df_filtered, n_total, n_kept = filter_min_history(df_long, "SKU", min_history_points=6)
    assert n_total == 2
    assert n_kept == 1
    assert df_filtered["SKU"].unique().tolist() == ["A"]


# ----------------------------------------------------------------------
# build_sku_series
# ----------------------------------------------------------------------

def test_build_sku_series_trims_leading_zeros():
    df = pd.DataFrame({
        "SKU": ["A"] * 5,
        "Date": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "Demand": [0, 0, 10, 20, 30],
    })
    series = build_sku_series(df, id_col="SKU", trim_leading_zeros=True)
    assert series["A"] == [10.0, 20.0, 30.0]


def test_build_sku_series_keeps_internal_zeros():
    df = pd.DataFrame({
        "SKU": ["A"] * 5,
        "Date": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "Demand": [10, 0, 20, 0, 30],   # zero interno deve restare
    })
    series = build_sku_series(df, id_col="SKU", trim_leading_zeros=True)
    assert series["A"] == [10.0, 0.0, 20.0, 0.0, 30.0]


def test_build_sku_series_keeps_trailing_zeros():
    df = pd.DataFrame({
        "SKU": ["A"] * 5,
        "Date": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "Demand": [10, 20, 30, 0, 0],
    })
    series = build_sku_series(df, id_col="SKU", trim_leading_zeros=True)
    assert series["A"] == [10.0, 20.0, 30.0, 0.0, 0.0]


def test_build_sku_series_no_trim():
    df = pd.DataFrame({
        "SKU": ["A"] * 5,
        "Date": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "Demand": [0, 0, 10, 20, 30],
    })
    series = build_sku_series(df, id_col="SKU", trim_leading_zeros=False)
    assert series["A"] == [0.0, 0.0, 10.0, 20.0, 30.0]


def test_build_sku_series_empty_after_trim_excluded():
    df = pd.DataFrame({
        "SKU": ["A"] * 3,
        "Date": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "Demand": [0, 0, 0],
    })
    series = build_sku_series(df, id_col="SKU", trim_leading_zeros=True)
    assert "A" not in series


# ----------------------------------------------------------------------
# build_backtest_series
# ----------------------------------------------------------------------

def test_build_backtest_series_splits_correctly():
    sku_series = {"A": list(range(20))}  # 0..19
    bt, actuals, skipped = build_backtest_series(
        sku_series, horizon_backtest=12, min_history_points=6
    )
    assert "A" in bt
    assert bt["A"] == list(range(8))               # primi 8 mesi (storico troncato)
    assert list(actuals["A"]) == list(range(8, 20))  # ultimi 12 mesi
    assert skipped == 0


def test_build_backtest_series_skips_too_short():
    # Serie di 10 mesi, horizon_backtest=12: troppo corta
    sku_series = {"A": list(range(10))}
    bt, actuals, skipped = build_backtest_series(
        sku_series, horizon_backtest=12, min_history_points=6
    )
    assert "A" not in bt
    assert skipped == 1


def test_build_backtest_series_skips_truncated_below_min_history():
    # Serie di 14 mesi, troncamento di 12 -> 2 mesi residui < 6 minimo
    sku_series = {"A": list(range(14))}
    bt, actuals, skipped = build_backtest_series(
        sku_series, horizon_backtest=12, min_history_points=6
    )
    assert "A" not in bt
    assert skipped == 1


# ----------------------------------------------------------------------
# apply_winsorize (integrato)
# ----------------------------------------------------------------------

def test_apply_winsorize_disabled_passes_through():
    df = pd.DataFrame({
        "SKU": ["A"] * 5,
        "Demand": [1.0, 100.0, 2.0, 3.0, 4.0],
    })
    out = apply_winsorize(df, "SKU", level=0.05, enabled=False)
    pd.testing.assert_series_equal(out["Demand"], df["Demand"])

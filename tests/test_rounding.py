"""
Test su round_to_pack: 3 modi (up/down/nearest), edge case
(pack=0/None, NaN, valori negativi).
"""

import math

import numpy as np
import pandas as pd

from forecast_lib.rounding import round_to_pack


# ----------------------------------------------------------------------
# Modi di arrotondamento base
# ----------------------------------------------------------------------

def test_nearest_rounds_up_when_closer_to_higher():
    # 13 / 6 = 2.166 -> nearest = 2 -> 12
    assert round_to_pack(13, 6, mode="nearest") == 12
    # 14 / 6 = 2.333 -> nearest = 2 -> 12
    assert round_to_pack(14, 6, mode="nearest") == 12
    # 15 / 6 = 2.500 -> banker's rounding di numpy: round(2.5) = 2 (verso pari)
    assert round_to_pack(15, 6, mode="nearest") == 12
    # 16 / 6 = 2.666 -> nearest = 3 -> 18
    assert round_to_pack(16, 6, mode="nearest") == 18


def test_up_always_rounds_up():
    assert round_to_pack(1, 6, mode="up") == 6
    assert round_to_pack(7, 6, mode="up") == 12
    assert round_to_pack(12, 6, mode="up") == 12  # esatto, non sale


def test_down_always_rounds_down():
    assert round_to_pack(11, 6, mode="down") == 6
    assert round_to_pack(6, 6, mode="down") == 6
    assert round_to_pack(5, 6, mode="down") == 0


# ----------------------------------------------------------------------
# Edge case sul pack
# ----------------------------------------------------------------------

def test_pack_none_only_rounds_decimals():
    assert round_to_pack(3.14159, None, decimals=2) == 3.14


def test_pack_zero_only_rounds_decimals():
    assert round_to_pack(3.14159, 0, decimals=2) == 3.14


def test_pack_negative_only_rounds_decimals():
    assert round_to_pack(3.14159, -5, decimals=2) == 3.14


def test_pack_one_just_rounds_to_decimals():
    # pack=1 -> equivalente all'arrotondamento ai decimali
    assert round_to_pack(3.14159, 1, mode="nearest", decimals=0) == 3
    assert round_to_pack(3.7, 1, mode="up", decimals=0) == 4


# ----------------------------------------------------------------------
# Edge case sul valore
# ----------------------------------------------------------------------

def test_nan_returns_nan():
    result = round_to_pack(float("nan"), 6)
    assert pd.isna(result)


def test_zero_value():
    assert round_to_pack(0, 6, mode="nearest") == 0
    assert round_to_pack(0, 6, mode="up") == 0
    assert round_to_pack(0, 6, mode="down") == 0


def test_value_exactly_a_pack_multiple_unchanged():
    assert round_to_pack(12, 6, mode="up") == 12
    assert round_to_pack(12, 6, mode="down") == 12
    assert round_to_pack(12, 6, mode="nearest") == 12


# ----------------------------------------------------------------------
# Decimals
# ----------------------------------------------------------------------

def test_decimals_applied_to_result():
    # 13 / 6 = 2.166... up -> 3 -> *6 = 18 esatto
    # caso con pack frazionario: 1.5 con pack 0.7 (innaturale ma valido)
    result = round_to_pack(2.5, 0.7, mode="nearest", decimals=2)
    # 2.5/0.7 = 3.571 -> nearest = 4 -> 4*0.7 = 2.8
    assert math.isclose(result, 2.8, rel_tol=1e-9)

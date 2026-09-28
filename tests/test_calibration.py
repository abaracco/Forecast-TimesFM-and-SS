"""
Test su theil_sen_log_trend e get_calibration_factor.

theil_sen_log_trend e' la funzione canonica del progetto: viene riusata
identica nel backtest. I test verificano il comportamento su:
  - serie esponenziale pura: deve recuperare beta atteso
  - serie troppo corta (<6 valori positivi): None, None
  - serie con zeri interni: deve filtrarli ma preservare le posizioni
"""

import math

import numpy as np

from forecast_lib.calibration import (
    get_calibration_factor,
    month_from_label,
    theil_sen_log_trend,
)


# ----------------------------------------------------------------------
# theil_sen_log_trend
# ----------------------------------------------------------------------

def test_pure_exponential_recovers_slope():
    # y = exp(0.1 * t) per t = 0..9 -> log(y) = 0.1 * t
    # beta atteso ~ 0.1, alpha atteso ~ 0
    t = np.arange(10)
    y = np.exp(0.1 * t)
    alpha, beta = theil_sen_log_trend(y.tolist())
    assert alpha is not None
    assert math.isclose(beta, 0.1, abs_tol=1e-9)
    assert math.isclose(alpha, 0.0, abs_tol=1e-9)


def test_constant_series_has_zero_slope():
    # serie costante -> beta = 0, alpha = log(costante)
    y = [10.0] * 10
    alpha, beta = theil_sen_log_trend(y)
    assert math.isclose(beta, 0.0, abs_tol=1e-9)
    assert math.isclose(alpha, math.log(10.0), abs_tol=1e-9)


def test_too_few_positives_returns_none():
    # solo 5 valori positivi -> non basta
    y = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    alpha, beta = theil_sen_log_trend(y)
    assert alpha is None and beta is None


def test_internal_zeros_are_filtered_but_positions_preserved():
    # se i positivi sono in posizioni 0,2,4,6,8,10 (con zeri interni),
    # il trend deve usare quelle posizioni temporali, non 0..5
    # y = [1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32]
    # log(positivi) = [0, log2, 2log2, 3log2, 4log2, 5log2] alle pos [0,2,4,6,8,10]
    # beta atteso = log(2) / 2  (raddoppia ogni 2 step)
    y = [1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32]
    alpha, beta = theil_sen_log_trend(y)
    assert alpha is not None
    expected_beta = math.log(2) / 2
    assert math.isclose(beta, expected_beta, rel_tol=1e-9)


def test_single_outlier_does_not_break_robustness():
    # serie quasi costante con un outlier: la mediana ignora l'outlier
    y = [10.0] * 9 + [10000.0]   # un outlier finale
    alpha, beta = theil_sen_log_trend(y)
    # robusto: la mediana delle pendenze e' ~0 (la maggior parte delle coppie
    # tra punti costanti hanno pendenza 0)
    assert math.isclose(beta, 0.0, abs_tol=1e-9)


# ----------------------------------------------------------------------
# month_from_label
# ----------------------------------------------------------------------

def test_month_from_label_valid():
    assert month_from_label("2024_05") == 5
    assert month_from_label("1999_12") == 12
    assert month_from_label("2024_01") == 1


def test_month_from_label_invalid_returns_none():
    assert month_from_label("bad") is None
    assert month_from_label("2024") is None


# ----------------------------------------------------------------------
# get_calibration_factor
# ----------------------------------------------------------------------

def test_calibration_disabled_returns_one():
    # CALIBRATION_MONTHS vuota -> sempre 1.0
    assert get_calibration_factor("SKU1", 8, {}, {}, []) == 1.0


def test_calibration_month_not_in_target_returns_one():
    # mese 5 non e' nei target [8, 12]
    sku_factors = {"SKU1": {8: 0.5, 12: 1.5}}
    global_factors = {8: 0.7, 12: 1.3}
    assert get_calibration_factor("SKU1", 5, sku_factors, global_factors, [8, 12]) == 1.0


def test_calibration_per_sku_takes_priority():
    sku_factors = {"SKU1": {8: 0.5}}
    global_factors = {8: 0.7}
    assert get_calibration_factor("SKU1", 8, sku_factors, global_factors, [8]) == 0.5


def test_calibration_falls_back_to_global_for_unknown_sku():
    sku_factors = {"SKU1": {8: 0.5}}
    global_factors = {8: 0.7}
    assert get_calibration_factor("SKU_UNKNOWN", 8, sku_factors, global_factors, [8]) == 0.7


def test_calibration_falls_back_to_one_if_no_data():
    # mese nei target ma nessun fattore disponibile
    assert get_calibration_factor("SKU1", 8, {}, {}, [8]) == 1.0

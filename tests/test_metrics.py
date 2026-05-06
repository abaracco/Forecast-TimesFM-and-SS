"""
Test sulla formula di accuratezza Motul.

Sono il cuore del backtest: la funzione DEVE essere immutabile
(requisito di business fisso). Qui copriamo:
  - caso perfetto (forecast = realta')
  - i 4 casi-zero della formula (ACT<=0, FCST<=0, sotto-stima >50%,
    sopra-stima >100%)
  - la versione weighted (peso = ACT + FCST)
"""

import math

from forecast_lib.metrics import accuracy_single_month, accuracy_weighted


# ----------------------------------------------------------------------
# accuracy_single_month
# ----------------------------------------------------------------------

def test_perfect_forecast_gives_one():
    assert accuracy_single_month(100, 100) == 1.0


def test_act_zero_gives_zero():
    assert accuracy_single_month(0, 50) == 0.0


def test_act_negative_gives_zero():
    assert accuracy_single_month(-10, 50) == 0.0


def test_fcst_zero_gives_zero():
    assert accuracy_single_month(100, 0) == 0.0


def test_fcst_negative_gives_zero():
    assert accuracy_single_month(100, -5) == 0.0


def test_under_forecast_more_than_half_gives_zero():
    # FCST < ACT/2  =>  delta = 60 > ACT(100)? no, 60 < 100
    # ma delta = 60 > FCST(40)? si' -> regola "sopra-stima > doppio" colpisce
    # Caso vero "sotto-stima > 50%": ACT=100, FCST=40 -> delta=60, delta>FCST(40) -> 0
    # Caso pulito: ACT=100, FCST=10  -> delta=90, delta < ACT(100) ok
    #                                   delta=90 > FCST(10)  -> 0  (sopra-stima inversa)
    # In pratica i due rami "delta>act" e "delta>fcst" coprono entrambi i lati.
    # ACT=100, FCST=49  -> delta=51, delta > FCST(49) -> 0
    assert accuracy_single_month(100, 49) == 0.0


def test_over_forecast_more_than_double_gives_zero():
    # ACT=100, FCST=201 -> delta=101, delta > ACT(100) -> 0
    assert accuracy_single_month(100, 201) == 0.0


def test_under_forecast_at_threshold_is_not_zero():
    # ACT=100, FCST=51 -> delta=49, delta<ACT(100), delta<FCST(51) -> ok
    # ACC = 1 - 49/100 = 0.51
    result = accuracy_single_month(100, 51)
    assert math.isclose(result, 0.51, rel_tol=1e-9)


def test_over_forecast_at_threshold_is_not_zero():
    # ACT=100, FCST=199 -> delta=99, delta<ACT(100), delta<FCST(199) -> ok
    # ACC = 1 - 99/100 = 0.01
    result = accuracy_single_month(100, 199)
    assert math.isclose(result, 0.01, rel_tol=1e-9)


def test_typical_case():
    # ACT=100, FCST=110 -> delta=10 -> ACC = 0.9
    result = accuracy_single_month(100, 110)
    assert math.isclose(result, 0.9, rel_tol=1e-9)


def test_accepts_floats():
    result = accuracy_single_month(100.5, 100.5)
    assert math.isclose(result, 1.0, rel_tol=1e-9)


# ----------------------------------------------------------------------
# accuracy_weighted
# ----------------------------------------------------------------------

def test_weighted_perfect_forecast():
    assert accuracy_weighted([100, 200, 50], [100, 200, 50]) == 1.0


def test_weighted_all_zero_weights_returns_zero():
    # se tutto e' 0/0, peso totale e' 0 -> ritorna 0
    assert accuracy_weighted([0, 0], [0, 0]) == 0.0


def test_weighted_more_volume_dominates():
    # Mese 1: ACT=1000, FCST=1000 (perfetto, peso 2000)
    # Mese 2: ACT=10, FCST=20 (acc=0.0 perche' sopra-doppio, peso 30)
    # Atteso: dominato dal mese 1
    result = accuracy_weighted([1000, 10], [1000, 20])
    # Numeratore: 1.0*2000 + 0.0*30 = 2000
    # Denominatore: 2030
    assert math.isclose(result, 2000.0 / 2030.0, rel_tol=1e-9)


def test_weighted_handles_per_month_zero_cases():
    # Mese 1: ok, Mese 2: ACT=0 -> ACC=0
    result = accuracy_weighted([100, 0], [100, 50])
    # Num: 1.0 * 200 + 0.0 * 50 = 200
    # Den: 250
    assert math.isclose(result, 200.0 / 250.0, rel_tol=1e-9)


def test_weighted_single_month_matches_single():
    # Con un solo mese, weighted == single (peso si semplifica)
    assert accuracy_weighted([100], [100]) == accuracy_single_month(100, 100)
    assert accuracy_weighted([100], [110]) == accuracy_single_month(100, 110)

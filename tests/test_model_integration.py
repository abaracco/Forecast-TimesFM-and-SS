"""
T1 — Equivalenza numerica e padding sul modello TimesFM REALE (marcato `slow`).

E' il test che sostiene tutto il resto del piano: se l'inferenza in batch non e'
equivalente a quella per singola serie, il guadagno di performance sarebbe
pagato in numeri diversi, e i gate di collaudo (§ 9) non avrebbero piu' una
baseline sensata.

Il modello viene caricato UNA volta per modulo e ricompilato a batch diversi con
`fl_recompile`, che e' l'unico modo di cambiare `global_batch_size` (si imposta
solo dentro `compile()`).

Prerequisiti: `./timesfm` gia' pinnato a v2.0.2 (ci pensa `setup_timesfm`), pesi
HuggingFace scaricabili o gia' in cache, e la CWD sulla root del repository.
Esecuzione: `pytest -m slow`.
"""

import numpy as np
import pytest

from forecast_lib.model import forecast_batch_with_fallback, setup_timesfm
from forecast_lib.rounding import round_to_pack

pytestmark = pytest.mark.slow


BATCH = 32          # lo stesso INFERENCE_BATCH_SIZE di default del Modulo A
HORIZON = 12
MIN_HISTORY_POINTS = 6


@pytest.fixture(scope="module")
def model():
    return setup_timesfm(
        colab=False,
        horizon=24,
        batch_size=BATCH,
        model_id="google/timesfm-2.5-200m-pytorch",
        model_revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        expected_version="2.0.2",
        timesfm_repo_url="https://github.com/google-research/timesfm.git",
        pin_strict=True,
    )


def shaped_series():
    """Le cinque forme di serie che la pipeline incontra davvero.

    Deterministiche (`RandomState(0)`): un test di equivalenza numerica su dati
    che cambiano a ogni esecuzione non sarebbe riproducibile.
    """
    rs = np.random.RandomState(0)
    base = rs.randint(50, 150, size=36).astype(np.float32)

    with_zeros = base.copy()
    with_zeros[[4, 9, 17]] = 0.0              # zeri interni: domanda vera, non NaN

    with_outliers = base.copy()
    with_outliers[12] = 5000.0
    with_outliers[25] = 1.0

    return {
        "corta": base[:MIN_HISTORY_POINTS].copy(),       # storico minimo ammesso
        "zeri_interni": with_zeros,
        "costante": np.full(24, 42.0, dtype=np.float32),
        "trend_forte": (np.arange(36) * 7 + 20).astype(np.float32),
        "outlier": with_outliers,
    }


def many_series(n, seed=1):
    rs = np.random.RandomState(seed)
    return [rs.randint(10, 500, size=rs.randint(8, 36)).astype(np.float32)
            for _ in range(n)]


def forecast_at(model, inputs, batch, horizon=HORIZON):
    """Ricompila al batch richiesto ed esegue il forecast. Nessun errore atteso:
    un fallimento qui sarebbe un problema di ambiente, non un caso da tollerare."""
    model.fl_recompile(batch)
    assert model.global_batch_size == batch
    results, errors = forecast_batch_with_fallback(model, inputs, horizon)
    assert errors == {}, f"forecast fallito al batch {batch}: {errors}"
    assert all(r is not None for r in results)
    return [np.asarray(r, dtype=float) for r in results]


# ----------------------------------------------------------------------
# T1.a — equivalenza numerica
# ----------------------------------------------------------------------

def test_batch_inference_matches_single_series_inference(model):
    series = shaped_series()
    inputs = list(series.values())

    at_1 = forecast_at(model, inputs, 1)
    at_n = forecast_at(model, inputs, BATCH)

    for name, one, many in zip(series, at_1, at_n):
        assert one.shape == (HORIZON,), name
        np.testing.assert_allclose(
            one, many, rtol=1e-4, atol=1e-3,
            err_msg=f"serie '{name}': batch {BATCH} diverge da batch 1",
        )


# ----------------------------------------------------------------------
# T1.b — equivalenza dopo l'arrotondamento consegnato
# ----------------------------------------------------------------------

def test_rounded_output_is_identical(model):
    """`pack=1`, `nearest`, 3 decimali: l'arrotondamento piu' fine che la
    pipeline possa applicare, quindi il criterio piu' severo."""
    inputs = list(shaped_series().values())

    at_1 = forecast_at(model, inputs, 1)
    at_n = forecast_at(model, inputs, BATCH)

    for one, many in zip(at_1, at_n):
        rounded_1 = [round_to_pack(v, 1, mode="nearest", decimals=3) for v in one]
        rounded_n = [round_to_pack(v, 1, mode="nearest", decimals=3) for v in many]
        assert rounded_1 == rounded_n


# ----------------------------------------------------------------------
# T1.c / T1.e — cardinalita' dell'output rispetto al batch
# ----------------------------------------------------------------------

def test_series_count_not_a_multiple_of_the_batch(model):
    inputs = many_series(65)
    results = forecast_at(model, inputs, BATCH)

    assert len(results) == 65
    assert all(r.shape == (HORIZON,) for r in results)


def test_fewer_series_than_the_batch(model):
    inputs = many_series(5, seed=2)

    at_n = forecast_at(model, inputs, BATCH)
    at_1 = forecast_at(model, inputs, 1)

    assert len(at_n) == 5
    for one, many in zip(at_1, at_n):
        np.testing.assert_allclose(one, many, rtol=1e-4, atol=1e-3)


# ----------------------------------------------------------------------
# T1.d — il padding non contamina l'ultimo chunk
# ----------------------------------------------------------------------

def test_padding_does_not_contaminate_the_last_chunk(model):
    """Con 33 serie e batch 32 l'ultimo chunk contiene UNA serie vera e 31 righe
    di padding. Le prime 32 sono un batch pieno e coinciderebbero banalmente:
    la riga che conta e' la 33esima."""
    inputs = many_series(33, seed=3)

    at_n = forecast_at(model, inputs, BATCH)
    assert len(at_n) == 33

    last = forecast_at(model, [inputs[32]], 1)[0]
    np.testing.assert_allclose(
        at_n[32], last, rtol=1e-4, atol=1e-3,
        err_msg="la 33esima serie e' contaminata dal padding dell'ultimo chunk",
    )
    # e non deve essere la previsione di una serie di zeri
    assert np.any(at_n[32] > 0)


# ----------------------------------------------------------------------
# T1.f — nessuna mutazione della lista del chiamante
# ----------------------------------------------------------------------

def test_the_caller_input_list_is_not_mutated(model):
    """TimesFM padda la lista in place: senza `list(inputs)` un retry sulla
    stessa lista restituirebbe righe fantasma. E' questo, piu' del conteggio
    delle righe, il test che protegge dal difetto."""
    inputs = many_series(5, seed=4)
    before = len(inputs)
    identities = [id(s) for s in inputs]

    model.fl_recompile(BATCH)
    forecast_batch_with_fallback(model, inputs, HORIZON)

    assert len(inputs) == before
    assert [id(s) for s in inputs] == identities

    # e un secondo giro sulla stessa lista resta corretto
    results, errors = forecast_batch_with_fallback(model, inputs, HORIZON)
    assert errors == {}
    assert len(results) == before
    assert len(inputs) == before


# ----------------------------------------------------------------------
# Contorno: stato del run
# ----------------------------------------------------------------------

def test_the_run_is_not_degraded_by_a_normal_forecast(model):
    inputs = many_series(4, seed=5)
    model.fl_recompile(BATCH)
    seconds_before = model.fl_inference_seconds

    forecast_batch_with_fallback(model, inputs, HORIZON)

    assert model.fl_degraded_after_inference is False
    assert model.fl_inference_seconds > seconds_before


def test_the_pin_is_verified_and_the_state_is_attached(model):
    assert model.fl_pin_verified is True
    assert model.fl_timesfm_tag == "v2.0.2"
    assert model.fl_model_revision == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    assert model.fl_batch_size_initial == BATCH

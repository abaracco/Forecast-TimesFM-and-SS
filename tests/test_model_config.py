"""
T3 — Configurazione del modello e percorsi degradati (veloce, tutto mockato).

Il modello vero non viene mai caricato: al suo posto il loader esegue davvero
(`spec_from_file_location` + `exec_module`) il finto `timesfm_2p5_torch.py` di
`tests/_fake_timesfm.py`. Il percorso di `setup_timesfm` e' quindi esercitato per
intero, ma senza rete e senza GPU.

Due scelte deliberate:

  - le asserzioni sulla configurazione si fanno sugli **argomenti passati al
    costruttore** di `ForecastConfig`, non sullo stato del modello: `compile()`
    riscrive `max_horizon` da 24 a 128, quindi rileggerlo dal modello proverebbe
    il contrario di quello che si vuole provare;
  - `torch` e' un doppio con una `OutOfMemoryError` propria: con il torch vero
    l'OOM non sarebbe provocabile a comando, e il ramo di degrado — la parte piu'
    delicata di `forecast_batch_with_fallback` — resterebbe non coperto.

Questo file e' anche il primo test che importa `model.py`: senza, un `pytest`
verde non direbbe nulla sul Modulo F.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from forecast_lib import model as fl_model
from forecast_lib.backtest import empty_backtest_results, run_backtest
from forecast_lib.export import save_audit_csvs
from forecast_lib.model import (
    _degradation_levels,
    _is_oom,
    forecast_all_skus_point,
    forecast_batch_with_fallback,
    setup_timesfm,
)

from tests._fake_timesfm import (
    HOOKS,
    REC,
    FakeForecastConfig,
    make_fake_torch,
    series_forecast,
    TORCH_MODULE_SOURCE,
)


PIN_INFO = {
    "path": "./timesfm",
    "tag": "v2.0.2",
    "pin_verified": True,
    "head": "0123456789abcdef",
    "action": "reused",
    "message": None,
}


class Env:
    """Ambiente di test: chiama `setup_timesfm` con i doppi gia' al loro posto."""

    def __init__(self, torch_module):
        self.torch = torch_module

    def setup(self, **overrides):
        kwargs = dict(
            colab=False,
            horizon=24,
            batch_size=32,
            model_id="fake/timesfm",
            model_revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
            expected_version="2.0.2",
            timesfm_repo_url="https://example.invalid/timesfm.git",
            pin_strict=True,
        )
        kwargs.update(overrides)
        return setup_timesfm(**kwargs)


@pytest.fixture
def env(tmp_path, monkeypatch):
    REC.reset()

    # 1. Struttura di file attesa dal loader, sotto una CWD temporanea:
    #    setup_timesfm usa "./timesfm" in locale, non un parametro.
    pkg_dir = tmp_path / "timesfm" / "src" / "timesfm" / "timesfm_2p5"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "timesfm_2p5_torch.py").write_text(TORCH_MODULE_SOURCE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # 2. Il pin e' gia' verificato: `ensure_timesfm_checkout` ha il suo test (T2b).
    monkeypatch.setattr(fl_model, "ensure_timesfm_checkout",
                        lambda *a, **k: dict(PIN_INFO))

    # 3. ForecastConfig: il loader la cerca con importlib.import_module.
    fake_pkg = type(sys)("timesfm")
    fake_cfg_mod = type(sys)("timesfm.config")
    fake_cfg_mod.ForecastConfig = FakeForecastConfig
    monkeypatch.setitem(sys.modules, "timesfm", fake_pkg)
    monkeypatch.setitem(sys.modules, "timesfm.config", fake_cfg_mod)

    # 4. torch doppio: device deterministico e OOM provocabile.
    torch_module = make_fake_torch(cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    # 5. La revision dei pesi si risolve via HuggingFace: qui non c'e' rete.
    monkeypatch.setattr(fl_model, "_resolve_model_revision",
                        lambda model_id, revision: revision)

    saved_path = list(sys.path)
    yield Env(torch_module)

    # `setup_timesfm` lascia dietro di se' il modulo caricato e la cartella
    # sorgente temporanea: vanno tolte, altrimenti il test successivo (e i test
    # `slow`, che caricano il modello vero) ereditano un sys.path/sys.modules
    # che punta a una tmp_path ormai cancellata.
    sys.modules.pop(fl_model.TORCH_MODULE_NAME, None)
    sys.path[:] = saved_path
    REC.reset()


# ======================================================================
# Configurazione: argomenti passati a ForecastConfig e a from_pretrained
# ======================================================================

def test_forecast_config_receives_the_expected_arguments(env):
    env.setup(horizon=24, batch_size=32)

    cfg = REC.config_kwargs[0]
    assert cfg["per_core_batch_size"] == 32
    assert cfg["max_context"] == 512
    assert cfg["max_horizon"] == 24
    assert cfg["normalize_inputs"] is True
    assert cfg["force_flip_invariance"] is True
    assert cfg["infer_is_positive"] is True
    assert cfg["fix_quantile_crossing"] is True


def test_batch_size_is_propagated_to_the_config(env):
    env.setup(batch_size=8)
    assert REC.config_kwargs[0]["per_core_batch_size"] == 8


def test_compile_rewrites_max_horizon_so_the_model_state_is_not_the_oracle(env):
    """Perche' T3 assevera sui kwargs del costruttore e non sul modello."""
    model = env.setup(horizon=24)
    assert REC.config_kwargs[0]["max_horizon"] == 24        # cio' che abbiamo chiesto
    assert model.forecast_config.max_horizon == 128         # cio' che resta dopo compile()


def test_model_revision_is_propagated_to_from_pretrained(env):
    env.setup(model_revision="deadbeefcafe")
    assert REC.from_pretrained == [("fake/timesfm", "deadbeefcafe")]


def test_model_revision_none_means_from_pretrained_without_revision(env):
    """Senza revision HuggingFace risolve 'main': la chiamata non deve passare
    `revision=None`, che avrebbe lo stesso effetto ma nasconderebbe il caso."""
    env.setup(model_revision=None)
    assert REC.from_pretrained == [("fake/timesfm", None)]


def test_run_state_attributes_are_attached_to_the_model(env):
    model = env.setup(batch_size=32)
    assert model.fl_batch_size == 32
    assert model.fl_batch_size_initial == 32
    assert model.fl_degraded is False
    assert model.fl_degraded_after_inference is False
    assert model.fl_device == "cpu"
    assert model.fl_model_revision == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    assert model.fl_pin_verified is True
    assert model.fl_timesfm_tag == "v2.0.2"
    assert callable(model.fl_recompile)


def test_pin_not_verified_is_propagated_to_the_model(env, monkeypatch):
    monkeypatch.setattr(fl_model, "ensure_timesfm_checkout",
                        lambda *a, **k: dict(PIN_INFO, pin_verified=False,
                                             message="pin non verificato"))
    model = env.setup()
    assert model.fl_pin_verified is False


def test_fl_recompile_rebuilds_the_original_config_except_the_batch_size(env):
    model = env.setup(horizon=24, batch_size=32)
    original = dict(REC.config_kwargs[0])

    model.fl_recompile(1)

    recompiled = dict(REC.config_kwargs[-1])
    assert recompiled["per_core_batch_size"] == 1
    assert original["per_core_batch_size"] == 32
    del recompiled["per_core_batch_size"]
    del original["per_core_batch_size"]
    # tutto il resto — max_horizon incluso, che compile() avrebbe portato a 128 —
    # deve essere identico all'originale
    assert recompiled == original
    assert model.fl_batch_size == 1
    assert model.global_batch_size == 1


def test_missing_torch_module_file_raises_explicitly(env, tmp_path, monkeypatch):
    (tmp_path / "timesfm" / "src" / "timesfm" / "timesfm_2p5"
     / "timesfm_2p5_torch.py").unlink()
    with pytest.raises(RuntimeError, match="Modulo TimesFM non trovato"):
        env.setup()


def test_missing_model_class_raises_explicitly(env, tmp_path):
    path = (tmp_path / "timesfm" / "src" / "timesfm" / "timesfm_2p5"
            / "timesfm_2p5_torch.py")
    path.write_text("class Altro:\n    pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="TimesFM_2p5_200M_torch"):
        env.setup()


def test_missing_forecast_config_raises_explicitly(env, monkeypatch):
    """Senza ForecastConfig il modello girerebbe con i default della libreria
    (batch 1, orizzonte diverso) senza dirlo: deve essere un errore."""
    monkeypatch.delitem(sys.modules, "timesfm.config")
    with pytest.raises(RuntimeError, match="ForecastConfig non trovata"):
        env.setup()


# ======================================================================
# Smoke test
# ======================================================================

def test_smoke_test_does_not_count_towards_inference_seconds(env):
    model = env.setup()
    # lo smoke test e' gia' avvenuto dentro setup_timesfm
    assert model.n_forecast_calls == 1
    assert model.fl_inference_seconds == 0.0

    forecast_batch_with_fallback(model, [np.arange(10, dtype=np.float32)], 6)
    assert model.fl_inference_seconds > 0.0


def test_smoke_test_failure_is_blocking(env):
    def always_fail(model, horizon, inputs):
        raise ValueError("modello rotto")

    HOOKS["forecast"] = always_fail
    with pytest.raises(RuntimeError, match="Smoke test TimesFM fallito"):
        env.setup()


def test_smoke_test_rejects_non_finite_output(monkeypatch, env):
    """Lo smoke test deve verificare che l'output sia sensato, non solo che la
    chiamata non sollevi: un modello che restituisce NaN e' inutilizzabile."""
    import tests._fake_timesfm as fake

    def nan_forecast(self, horizon, inputs):
        return np.full((len(inputs), horizon), np.nan), None

    monkeypatch.setattr(fake.FakeTimesFM, "forecast", nan_forecast)
    with pytest.raises(RuntimeError, match="output non finito"):
        env.setup()


def test_smoke_test_rejects_unexpected_output_shape(monkeypatch, env):
    import tests._fake_timesfm as fake

    def short_forecast(self, horizon, inputs):
        return np.zeros((len(inputs), horizon - 1)), None

    monkeypatch.setattr(fake.FakeTimesFM, "forecast", short_forecast)
    with pytest.raises(RuntimeError, match="forma inattesa"):
        env.setup()


def test_oom_during_smoke_test_does_not_flag_the_run_as_unusable(env):
    """Un OOM allo smoke test avviene PRIMA di qualunque inferenza reale: il run
    gira tutto al batch degradato ed e' pienamente consegnabile."""
    def oom_above_8(model, horizon, inputs):
        if model.global_batch_size > 8:
            raise env.torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    HOOKS["forecast"] = oom_above_8
    model = env.setup(batch_size=32)

    assert model.fl_batch_size == 8
    assert model.fl_degraded is True
    assert model.fl_degraded_after_inference is False
    assert model.fl_inference_seconds == 0.0


# ======================================================================
# forecast_batch_with_fallback — i tre livelli
# ======================================================================

def _series(n_series=3, length=10):
    return [np.arange(i + 1, i + 1 + length, dtype=np.float32)
            for i in range(n_series)]


def test_happy_path_returns_one_row_per_input_in_order(env):
    model = env.setup(batch_size=32)
    inputs = _series(3)

    results, errors = forecast_batch_with_fallback(model, inputs, 6)

    assert errors == {}
    assert len(results) == 3
    for i, series in enumerate(inputs):
        np.testing.assert_allclose(results[i], series_forecast(series, 6))


def test_empty_input_list_does_not_call_the_model(env):
    model = env.setup()
    calls_before = model.n_forecast_calls
    results, errors = forecast_batch_with_fallback(model, [], 6)
    assert results == [] and errors == {}
    assert model.n_forecast_calls == calls_before


def test_caller_input_list_is_never_mutated(env):
    """La libreria vera padda la lista in place: senza `list(inputs)` un retry
    restituirebbe righe fantasma."""
    model = env.setup(batch_size=32)
    inputs = _series(3)

    forecast_batch_with_fallback(model, inputs, 6)
    assert len(inputs) == 3

    # e un secondo giro sulla stessa lista resta corretto
    results, _ = forecast_batch_with_fallback(model, inputs, 6)
    assert len(inputs) == 3
    assert len(results) == 3


def test_oom_degrades_along_the_derived_scale(env):
    model = env.setup(batch_size=32)
    seen_batches = []

    def oom_until_batch_1(m, horizon, inputs):
        seen_batches.append(m.global_batch_size)
        if m.global_batch_size > 1:
            raise env.torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    model.behavior = oom_until_batch_1
    inputs = _series(3)
    results, errors = forecast_batch_with_fallback(model, inputs, 6)

    assert seen_batches == [32, 8, 1]          # scala [N, N//4, 1]
    assert errors == {}
    for i, series in enumerate(inputs):
        np.testing.assert_allclose(results[i], series_forecast(series, 6))
    assert model.fl_degraded is True
    assert model.fl_degraded_after_inference is True
    assert model.fl_batch_size == 1


def test_oom_stops_at_the_first_level_that_works(env):
    model = env.setup(batch_size=32)
    seen_batches = []

    def oom_above_8(m, horizon, inputs):
        seen_batches.append(m.global_batch_size)
        if m.global_batch_size > 8:
            raise env.torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    model.behavior = oom_above_8
    results, errors = forecast_batch_with_fallback(model, _series(3), 6)

    assert seen_batches == [32, 8]
    assert errors == {}
    assert model.fl_batch_size == 8


def test_oom_frees_cuda_memory_before_retrying(env):
    model = env.setup(batch_size=32)

    def oom_above_1(m, horizon, inputs):
        if m.global_batch_size > 1:
            raise env.torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    model.behavior = oom_above_1
    # con cuda non disponibile empty_cache non viene chiamata: si verifica che
    # il percorso sia comunque percorso senza sollevare
    env.torch.cuda.is_available = lambda: True
    forecast_batch_with_fallback(model, _series(3), 6)
    assert env.torch.fl_calls["empty_cache"] >= 2


def test_non_oom_failure_degrades_to_batch_1_before_the_per_input_loop(env):
    """Un loop per-input al batch 32 farebbe 32 serie di calcolo per ogni
    risultato utile: il degrado a 1 deve avvenire PRIMA del loop."""
    model = env.setup(batch_size=32)
    seen = []

    def fail_in_batch(m, horizon, inputs):
        seen.append((m.global_batch_size, len(inputs)))
        if len(inputs) > 1:
            raise ValueError("errore non-OOM")

    model.behavior = fail_in_batch
    inputs = _series(3)
    results, errors = forecast_batch_with_fallback(model, inputs, 6)

    assert seen[0] == (32, 3)                       # tentativo in batch
    assert [s[0] for s in seen[1:]] == [1, 1, 1]    # loop gia' a batch 1
    assert errors == {}
    for i, series in enumerate(inputs):
        np.testing.assert_allclose(results[i], series_forecast(series, 6))
    assert model.fl_batch_size == 1
    assert model.fl_degraded is True
    assert model.fl_degraded_after_inference is True


def test_non_oom_failure_does_not_walk_the_oom_scale(env):
    """Il degrado intermedio serve solo all'OOM: su un errore diverso si passa
    direttamente al livello 3."""
    model = env.setup(batch_size=32)
    seen = []

    def fail_in_batch(m, horizon, inputs):
        seen.append(m.global_batch_size)
        if len(inputs) > 1:
            raise ValueError("errore non-OOM")

    model.behavior = fail_in_batch
    forecast_batch_with_fallback(model, _series(3), 6)
    assert 8 not in seen


def test_per_input_failures_are_reported_by_index(env):
    model = env.setup(batch_size=32)
    inputs = _series(3)

    def fail_batch_and_second_series(m, horizon, ins):
        if len(ins) > 1:
            raise ValueError("errore non-OOM")
        if float(np.sum(ins[0])) == float(np.sum(inputs[1])):
            raise ValueError("questa serie non si prevede")

    model.behavior = fail_batch_and_second_series
    results, errors = forecast_batch_with_fallback(model, inputs, 6)

    assert set(errors) == {1}
    assert "questa serie non si prevede" in errors[1]
    assert results[1] is None
    np.testing.assert_allclose(results[0], series_forecast(inputs[0], 6))
    np.testing.assert_allclose(results[2], series_forecast(inputs[2], 6))


def test_degradation_is_permanent_within_the_run(env):
    model = env.setup(batch_size=32)
    calls = {"n": 0}

    def oom_once(m, horizon, inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise env.torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

    model.behavior = oom_once
    forecast_batch_with_fallback(model, _series(3), 6)
    assert model.fl_batch_size == 8

    # seconda chiamata: si riparte dal livello degradato, non da 32
    REC.forecast_calls.clear()
    forecast_batch_with_fallback(model, _series(3), 6)
    assert [c["batch"] for c in REC.forecast_calls] == [8]


def test_forecast_all_skus_point_maps_results_and_errors_by_sku(env):
    model = env.setup(batch_size=32)
    series_dict = {
        "A": [1, 2, 3, 4, 5, 6],
        "B": [10, 20, 30, 40, 50, 60],
        "C": [7, 7, 7, 7, 7, 7],
    }

    def fail_batch_and_sku_b(m, horizon, ins):
        if len(ins) > 1:
            raise ValueError("errore non-OOM")
        if float(np.sum(ins[0])) == 210.0:      # SKU B
            raise ValueError("serie B non prevedibile")

    model.behavior = fail_batch_and_sku_b
    results, errors = forecast_all_skus_point(model, series_dict, 6)

    assert set(results) == {"A", "C"}
    assert set(errors) == {"B"}
    np.testing.assert_allclose(
        results["A"], series_forecast(np.array(series_dict["A"]), 6))


# ======================================================================
# Funzioni pure di supporto al degrado
# ======================================================================

def test_degradation_levels_default_scale():
    assert _degradation_levels(32, 32) == [32, 8, 1]


def test_degradation_levels_never_climb_back_up():
    # il degrado e' permanente: i livelli gia' superati vanno scartati
    assert _degradation_levels(32, 8) == [8, 1]
    assert _degradation_levels(32, 1) == [1]


def test_degradation_levels_are_deduplicated_for_small_batches():
    assert _degradation_levels(4, 4) == [4, 1]
    assert _degradation_levels(2, 2) == [2, 1]
    assert _degradation_levels(1, 1) == [1]


def test_is_oom_recognises_the_typed_error(env):
    assert _is_oom(env.torch.cuda.OutOfMemoryError("boom")) is True


def test_is_oom_recognises_the_runtime_error_by_message(env):
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate...")) is True


def test_is_oom_ignores_unrelated_errors(env):
    assert _is_oom(ValueError("niente a che vedere")) is False
    assert _is_oom(RuntimeError("shape mismatch")) is False


# ======================================================================
# run_backtest — q_global e SKU falliti
# ======================================================================

def _backtest_frame(skus, n_months=30):
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    rows = []
    for offset, sku in enumerate(skus):
        for i, date in enumerate(dates):
            rows.append({
                "SKU": sku,
                "Description": f"desc {sku}",
                "Round": 1,
                "BUn": "EA",
                "Date": date,
                "Period": date.strftime("%Y_%m"),
                "Demand": 100 + offset * 10 + (i % 5),
            })
    df = pd.DataFrame(rows)
    series = {
        sku: df[df["SKU"] == sku].sort_values("Date")["Demand"].tolist()
        for sku in skus
    }
    return df, series


def _run_backtest(model, df, series, **overrides):
    kwargs = dict(
        id_col="SKU",
        pack_size_col="Round",
        uom_col="BUn",
        horizon_backtest=6,
        min_history_points=6,
        n_backtest_origins=1,
        quantile_grid=[0.10, 0.30, 0.50, 0.70, 0.90],
        calibration_months=[8, 12],
        rounding_mode="nearest",
        round_decimals=3,
        shrinkage_enabled=True,
    )
    kwargs.update(overrides)
    return run_backtest(model, series, df, **kwargs)


def test_run_backtest_exposes_q_global_in_attrs(env):
    model = env.setup(batch_size=32)
    df, series = _backtest_frame(["A", "B", "C"])

    result = _run_backtest(model, df, series)

    assert len(result) == 3
    assert isinstance(result.attrs["q_global"], float)
    assert result.attrs["n_backtest_skus"] == 3
    assert result.attrs["n_skus_excluded"] == 0
    assert result.attrs["n_skus_zero_accuracy"] == int(
        (result["BestAccuracyRaw"] <= 0).sum())
    assert "BestQuantileRaw" in result.columns
    assert "BestAccuracyRaw" in result.columns


def test_run_backtest_without_shrinkage_leaves_q_global_none(env):
    model = env.setup(batch_size=32)
    df, series = _backtest_frame(["A", "B"])

    result = _run_backtest(model, df, series, shrinkage_enabled=False)

    assert result.attrs["q_global"] is None
    # senza shrinkage i valori grezzi coincidono con quelli finali
    assert (result["BestQuantile"] == result["BestQuantileRaw"]).all()
    assert (result["BestAccuracy"] == result["BestAccuracyRaw"]).all()


def test_run_backtest_with_no_eligible_sku_does_not_raise(env):
    """Storico piu' corto della finestra di backtest: nessun SKU valutabile."""
    model = env.setup(batch_size=32)
    df, series = _backtest_frame(["A"], n_months=5)

    result = _run_backtest(model, df, series)

    assert len(result) == 0
    assert result.attrs["q_global"] is None
    assert result.attrs["n_backtest_skus"] == 0


def test_run_backtest_drops_skus_whose_forecast_failed(env):
    """Uno SKU senza forecast sparisce dai risultati: nel Modulo H ricade su
    `best_q_map.get(sku, 0.5)`. Non deve far esplodere nulla."""
    model = env.setup(batch_size=32)
    df, series = _backtest_frame(["A", "B", "C"])
    failing_sum = float(np.sum(series["B"][:-6]))

    def fail_batch_and_sku_b(m, horizon, ins):
        if len(ins) > 1:
            raise ValueError("errore non-OOM")
        if float(np.sum(ins[0])) == failing_sum:
            raise ValueError("serie B non prevedibile")

    model.behavior = fail_batch_and_sku_b
    result = _run_backtest(model, df, series)

    assert set(result["SKU"]) == {"A", "C"}
    assert result.attrs["n_backtest_skus"] == 2
    assert result.attrs["n_skus_excluded"] == 1
    best_q_map = dict(zip(result["SKU"], result["BestQuantile"]))
    assert best_q_map.get("B", 0.5) == 0.5


def test_empty_backtest_results_has_the_full_schema():
    df = empty_backtest_results()
    assert list(df.columns) == ["SKU", "BestQuantile", "BestQuantileRaw",
                                "BestAccuracy", "BestAccuracyRaw", "TotalWeight"]
    assert df.attrs["q_global"] is None
    assert df.attrs["n_backtest_skus"] == 0
    assert df.attrs["n_skus_excluded"] == 0
    assert df.attrs["n_skus_zero_accuracy"] == 0


def test_audit_csv_materialises_q_global_as_a_column(env, tmp_path):
    """`df.attrs` non sopravvive a `to_csv`: la diagnostica di collaudo legge
    `q_global` dalla colonna."""
    model = env.setup(batch_size=32)
    df, series = _backtest_frame(["A", "B"])
    result = _run_backtest(model, df, series)

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    paths = save_audit_csvs(result, {"X": "errore"}, str(out_dir), "Forecast", " 2026 01 01")

    backtest_csv = pd.read_csv(paths[0])
    assert "q_global" in backtest_csv.columns
    assert backtest_csv["q_global"].nunique() == 1
    assert backtest_csv["q_global"].iloc[0] == pytest.approx(result.attrs["q_global"])

    errors_csv = pd.read_csv(paths[1])
    assert list(errors_csv.columns) == ["SKU", "Errore"]
    assert errors_csv["SKU"].tolist() == ["X"]


def test_audit_csv_survives_a_missing_q_global(env, tmp_path):
    df = empty_backtest_results()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    paths = save_audit_csvs(df, {}, str(out_dir), "Forecast", " 2026 01 01")

    backtest_csv = pd.read_csv(paths[0])
    assert "q_global" in backtest_csv.columns
    assert len(backtest_csv) == 0

"""
Doppio di TimesFM per i test veloci (T3).

Non e' un test: e' il modulo che `test_model_config.py` fa eseguire a
`spec_from_file_location` al posto di `timesfm_2p5_torch.py`. Il nome inizia con
'_' cosi' pytest non prova a raccoglierlo.

Il fake riproduce le tre proprieta' della libreria vera da cui dipende il codice
di produzione:

  1. `compile(forecast_config)` e' l'UNICO punto in cui si imposta
     `global_batch_size`, e riscrive `max_horizon` a 128 sulla config ricevuta
     (per questo le asserzioni di T3 vanno fatte sugli argomenti passati al
     costruttore, non sullo stato del modello);
  2. `forecast(horizon, inputs)` **muta in place** la lista ricevuta, paddandola
     fino a un multiplo di `global_batch_size`, e restituisce solo le prime
     `len(inputs)` righe (timesfm_2p5_base.py:166-168);
  3. il forecast e' deterministico per serie, cosi' ogni riga di output e'
     riconducibile al proprio input e il padding non puo' passare inosservato.
"""

import types

import numpy as np


class Recorder:
    """Raccoglie le chiamate del fake: e' su questo che T3 fa le asserzioni."""

    def __init__(self):
        self.reset()

    def reset(self):
        HOOKS["forecast"] = None
        self.config_kwargs = []      # kwargs passati al costruttore ForecastConfig
        self.compiles = []           # oggetti config passati a compile()
        self.from_pretrained = []    # (model_id, revision)
        self.forecast_calls = []     # {"horizon", "n_inputs", "batch"}


# Gancio globale sul forecast: serve quando il comportamento va imposto PRIMA
# che il modello esista, cioe' allo smoke test dentro `setup_timesfm`.
HOOKS = {"forecast": None}

REC = Recorder()


class FakeForecastConfig:
    """Sostituto di timesfm.ForecastConfig: registra i kwargs del costruttore."""

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        REC.config_kwargs.append(dict(kwargs))

    def __repr__(self):
        return f"FakeForecastConfig({self.kwargs})"


def series_forecast(series, horizon):
    """Forecast deterministico e distinguibile: media della serie + indice.

    La media lega ogni riga di output al proprio input (uno scambio di righe o
    una riga di padding vengono riconosciuti dal test) e resta dello stesso
    ordine di grandezza dello storico, cosi' i test che passano dal backtest
    vero producono accuratezze Motul sensate invece che nulle ovunque.
    """
    arr = np.asarray(series, dtype=float)
    base = float(arr.mean()) if arr.size else 0.0
    return base + np.arange(horizon, dtype=float)


class FakeTimesFM:
    """Doppio della classe TimesFM_2p5_200M_torch."""

    def __init__(self, model_id=None, revision=None):
        self.model_id = model_id
        self.revision = revision
        self.forecast_config = None
        self.global_batch_size = None
        self.device = None
        self.evaluated = False
        # Gancio per i test: callable(model, horizon, inputs); puo' sollevare.
        self.behavior = None
        self.n_forecast_calls = 0

    @classmethod
    def from_pretrained(cls, model_id, revision=None):
        REC.from_pretrained.append((model_id, revision))
        return cls(model_id=model_id, revision=revision)

    def compile(self, forecast_config):
        REC.compiles.append(forecast_config)
        self.forecast_config = forecast_config
        self.global_batch_size = forecast_config.per_core_batch_size
        # La libreria vera riscrive l'orizzonte con il massimo supportato.
        forecast_config.max_horizon = 128

    def to(self, device):
        self.device = device

    def eval(self):
        self.evaluated = True

    def forecast(self, horizon, inputs):
        self.n_forecast_calls += 1
        REC.forecast_calls.append({
            "horizon": horizon,
            "n_inputs": len(inputs),
            "batch": self.global_batch_size,
        })

        hook = self.behavior if self.behavior is not None else HOOKS["forecast"]
        if hook is not None:
            hook(self, horizon, inputs)

        num_inputs = len(inputs)
        batch = self.global_batch_size or 1
        missing = (-num_inputs) % batch
        if missing:
            # Mutazione in place della lista del chiamante, come la libreria vera.
            inputs += [np.array([0.0, 0.0, 0.0], dtype=np.float32)] * missing

        points = np.array([series_forecast(s, horizon) for s in inputs], dtype=float)
        quantiles = np.zeros((len(inputs), horizon, 10), dtype=float)
        return points[:num_inputs], quantiles[:num_inputs]


# Contenuto del finto `timesfm_2p5_torch.py`: il loader lo esegue davvero con
# `spec_from_file_location` + `exec_module`, quindi il percorso di import di
# `model.setup_timesfm` viene esercitato per intero.
TORCH_MODULE_SOURCE = """\
from tests._fake_timesfm import FakeTimesFM


class TimesFM_2p5_200M_torch(FakeTimesFM):
    pass
"""


def make_fake_torch(cuda_available=False, device_name="FakeGPU"):
    """Modulo `torch` minimale: device detection, empty_cache, OutOfMemoryError.

    Evita di dipendere dalla presenza (e dal device) del torch reale, e
    soprattutto rende deterministico il ramo OOM, che con il torch vero non
    sarebbe provocabile a comando.
    """

    class OutOfMemoryError(RuntimeError):
        pass

    calls = {"empty_cache": 0}

    def empty_cache():
        calls["empty_cache"] += 1

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda *a, **k: device_name,
        empty_cache=empty_cache,
        OutOfMemoryError=OutOfMemoryError,
    )
    torch.fl_calls = calls
    return torch


def raise_oom(torch_module, message="CUDA out of memory (fake)"):
    """Solleva l'eccezione OOM tipizzata del fake torch."""
    raise torch_module.cuda.OutOfMemoryError(message)

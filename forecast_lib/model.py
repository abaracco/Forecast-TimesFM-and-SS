"""
Modulo F — Caricamento modello TimesFM e funzione di forecast batch.

Il loader fa diverse cose, in ordine:
  1. Imposta la cache HuggingFace (effimera in Colab, persistente in locale)
  2. Porta ./timesfm al tag pinnato e ne verifica il pin (versioning.py)
  3. Aggiunge la cartella sorgente a sys.path
  4. Importa il modulo PyTorch da un percorso esplicito tramite
     spec_from_file_location (evita conflitti col pacchetto pip "timesfm",
     che non sempre e' compatibile con Python 3.12 di Colab)
  5. Preleva la classe modello per nome
  6. Scarica i pesi pre-addestrati da HuggingFace alla revision pinnata e ne
     risolve l'hash effettivo
  7. Configura ForecastConfig (max_context, max_horizon, per_core_batch_size)
  8. Sposta il modello su GPU se disponibile, altrimenti CPU
  9. Esegue uno smoke test attraverso lo stesso percorso di inferenza della
     pipeline; se fallisce, SOLLEVA

Al modello vengono attaccati alcuni attributi `fl_*` (stato del run:
batch size, degrado, device, revision, tempo di inferenza, esito del pin) e la
closure `fl_recompile`. TimesFM_2p5 e' una classe normale, senza __slots__ e
non dataclass: iniettarli e' lecito, e il prefisso evita collisioni. La firma
di `setup_timesfm` resta "restituisce il modello", cosi' i call site del
notebook e di backtest.py non cambiano.

`forecast_batch_with_fallback` e' il punto unico di ingresso all'inferenza:
un tentativo in batch, degrado su OOM, e in ultima istanza un giro per singolo
input. Sia `forecast_all_skus_point` che `backtest.run_backtest` passano di li'.
"""

import gc
import importlib
import importlib.util
import os
import pathlib
import sys
import time

import numpy as np

from .versioning import ensure_timesfm_checkout, timesfm_tag


# Percorso del modulo torch dentro il checkout TimesFM (verificato a v2.0.2).
TORCH_MODULE_RELPATH = ("src", "timesfm", "timesfm_2p5", "timesfm_2p5_torch.py")
TORCH_MODULE_NAME = "timesfm.timesfm_2p5.timesfm_2p5_torch"
MODEL_CLASS_NAME = "TimesFM_2p5_200M_torch"


def setup_timesfm(
    *,
    colab,
    horizon,
    batch_size=32,
    model_id="google/timesfm-2.5-200m-pytorch",
    model_revision=None,
    expected_version="2.0.2",
    timesfm_repo_url="https://github.com/google-research/timesfm.git",
    pin_strict=True,
):
    """
    Carica il modello TimesFM gestendo Colab e locale in modo trasparente.

    Restituisce l'istanza del modello compilata e pronta all'inferenza, con gli
    attributi `fl_*` descritti nel docstring del modulo.

    Parametri:
        colab:             True per modalita' Colab, False per locale
        horizon:           orizzonte massimo di forecast (passato a ForecastConfig)
        batch_size:        serie inviate insieme al modello a ogni passata, per
                           dispositivo (INFERENCE_BATCH_SIZE). Il default della
                           libreria e' 1, cioe' una serie alla volta
        model_id:          nome del modello su HuggingFace
        model_revision:    revision dei pesi (commit hash). None = 'main', cioe'
                           pesi che possono cambiare senza lasciare traccia
        expected_version:  versione TimesFM da pinnare, senza la 'v' iniziale
        timesfm_repo_url:  URL del repository TimesFM
        pin_strict:        True = un pin non verificabile blocca l'esecuzione
    """
    # 1. Cache HuggingFace
    if colab:
        os.environ.setdefault("HF_HOME", "/content/.cache/huggingface")
    # In locale: HF_HOME non viene impostato, usa il default (~/.cache/huggingface).

    # 2. Checkout TimesFM pinnato e verificato
    timesfm_dir = "/content/timesfm" if colab else "./timesfm"
    tag = timesfm_tag(expected_version)
    pin_info = ensure_timesfm_checkout(
        timesfm_dir, timesfm_repo_url, tag, strict=pin_strict
    )

    # 3. Aggiungi la cartella sorgente a sys.path
    pkg_dir = os.path.join(timesfm_dir, "src")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    # 4. Import del modulo PyTorch da percorso esplicito
    torch_mod_path = pathlib.Path(timesfm_dir).joinpath(*TORCH_MODULE_RELPATH)
    if not torch_mod_path.is_file():
        raise RuntimeError(
            f"Modulo TimesFM non trovato: '{torch_mod_path}'. Il checkout in "
            f"'{timesfm_dir}' non ha la struttura attesa per {tag}."
        )
    print("Modulo TimesFM:", torch_mod_path)

    spec = importlib.util.spec_from_file_location(TORCH_MODULE_NAME, str(torch_mod_path))
    torch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch_mod)
    sys.modules[TORCH_MODULE_NAME] = torch_mod

    # 5. Classe modello, per nome
    ModelClass = getattr(torch_mod, MODEL_CLASS_NAME, None)
    if ModelClass is None:
        raise RuntimeError(
            f"Classe '{MODEL_CLASS_NAME}' non trovata in '{torch_mod_path}'. "
            f"Il modulo TimesFM a {tag} non ha la struttura attesa."
        )
    print("Classe modello:", ModelClass)

    # 6. Pesi pre-addestrati.
    #    La revision effettiva non e' esposta ne' da from_pretrained ne' da
    #    _from_pretrained: la si risolve a parte dal path dello snapshot.
    resolved_revision = _resolve_model_revision(model_id, model_revision)
    print(f"Download pesi (HuggingFace): {model_id} @ {model_revision or 'main'}")
    if model_revision is None:
        model = ModelClass.from_pretrained(model_id)
    else:
        model = ModelClass.from_pretrained(model_id, revision=model_revision)
    if resolved_revision:
        print(f"Revision pesi risolta: {resolved_revision}")

    # 7. Configurazione parametri di inferenza
    ForecastConfig = _import_forecast_config()
    cfg_kwargs = dict(
        max_context=512,
        max_horizon=horizon,
        normalize_inputs=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )

    def _fl_recompile(new_batch_size):
        """Ricompila il modello a un batch size diverso.

        Ricostruisce la ForecastConfig dai kwargs originali invece di derivarla
        da quella in uso: dopo `compile()` quest'ultima ha max_horizon = 128.
        `global_batch_size` viene impostato solo dentro `compile()`, quindi ogni
        cambio di batch DEVE passare da qui.
        """
        new_cfg = ForecastConfig(per_core_batch_size=new_batch_size, **cfg_kwargs)
        model.compile(new_cfg)
        model.fl_batch_size = new_batch_size
        return new_cfg

    model.compile(ForecastConfig(per_core_batch_size=batch_size, **cfg_kwargs))

    # 8. Rilevamento automatico CPU/GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = device
    if device == "cuda":
        try:
            device_label = f"cuda ({torch.cuda.get_device_name()})"
        except Exception:
            pass
    print(f"Device: {device_label}")

    # TimesFM non deriva da torch.nn.Module -> .to() e .eval() opzionali
    if hasattr(model, "to"):
        try:
            model.to(device)
            print("Modello spostato su device tramite .to()")
        except Exception:
            print(".to() disponibile ma non utilizzabile - ignorato.")
    else:
        print("Modello senza .to(): TimesFM gestisce automaticamente CPU/GPU.")

    if hasattr(model, "eval"):
        try:
            model.eval()
            print("Modalita' eval attivata")
        except Exception:
            print(".eval() disponibile ma non chiamabile - ignorato.")
    else:
        print("Modello senza .eval(): TimesFM e' gia' in modalita' inferenza.")

    # Stato del run allegato al modello
    model.fl_recompile = _fl_recompile
    model.fl_batch_size = batch_size
    model.fl_batch_size_initial = batch_size
    model.fl_degraded = False
    model.fl_degraded_after_inference = False
    model.fl_has_produced_output = False
    model.fl_device = device_label
    model.fl_model_revision = resolved_revision
    model.fl_inference_seconds = 0.0
    model.fl_pin_verified = pin_info["pin_verified"]
    model.fl_timesfm_tag = pin_info["tag"]

    print("Modello pronto (loader manuale TimesFM attivato).")
    print(f"Batch size di inferenza: {batch_size} "
          f"(global_batch_size = {getattr(model, 'global_batch_size', 'n/d')})")

    # 9. Smoke test — attraverso lo stesso percorso della pipeline
    _smoke_test(model)

    return model


def _import_forecast_config():
    """Restituisce la classe ForecastConfig del checkout TimesFM caricato.

    Solleva se non la trova: senza, il modello girerebbe con la configurazione
    di default della libreria (batch 1, orizzonte diverso) senza dirlo.
    """
    tried = []
    for module_path in (
        "timesfm.config",
        "timesfm.configs",
        "timesfm.timesfm_2p5.configs.forecast_config",
    ):
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            tried.append(f"{module_path}: {exc}")
            continue
        if hasattr(mod, "ForecastConfig"):
            return mod.ForecastConfig
        tried.append(f"{module_path}: nessuna ForecastConfig")

    raise RuntimeError(
        "ForecastConfig non trovata nel checkout TimesFM. Tentativi:\n  "
        + "\n  ".join(tried)
    )


def _resolve_model_revision(model_id, model_revision):
    """Hash della revision dei pesi effettivamente usata.

    Ne' `from_pretrained` ne' `_from_pretrained` espongono il path dello
    snapshot: lo si ottiene scaricando un file leggero e leggendo l'hash dal
    percorso `snapshots/<hash>/`. Diagnostica, quindi non bloccante.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "config.json", revision=model_revision)
        parts = pathlib.Path(path).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
        return None
    except Exception as exc:
        print(f"Revision dei pesi non risolta ({exc}).")
        return None


def _smoke_test(model):
    """Verifica che il modello risponda, passando dal percorso di inferenza reale.

    Con un batch size > 1 questa singola serie attiva subito il ramo di padding
    interno di TimesFM: e' il canary del percorso nuovo. `count_time=False`
    tiene lo smoke test fuori da `fl_inference_seconds`, che a batch 32
    conterebbe 32 serie di calcolo per una sola serie utile.

    Solleva se fallisce anche dopo tutti i livelli di fallback: un modello che
    non risponde qui non produrra' nulla di utilizzabile dopo.
    """
    test = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19], dtype=np.float32)
    preds, errors = forecast_batch_with_fallback(model, [test], 3, count_time=False)

    if preds[0] is None:
        raise RuntimeError(f"Smoke test TimesFM fallito: {errors.get(0, 'errore ignoto')}")

    arr = np.asarray(preds[0], dtype=float)
    if arr.shape != (3,):
        raise RuntimeError(
            f"Smoke test TimesFM: forma inattesa dell'output {arr.shape}, attesa (3,)."
        )
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Smoke test TimesFM: output non finito {arr}.")

    print("Smoke test OK ->", arr)


# ----------------------------------------------------------------------
# Inferenza con fallback
# ----------------------------------------------------------------------

def _is_oom(exc):
    """True se l'eccezione e' un esaurimento di memoria GPU."""
    torch = sys.modules.get("torch")
    if torch is not None:
        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_cuda_memory():
    """Libera la memoria GPU. Va chiamata FUORI dal blocco except: dentro,
    l'eccezione tiene vivo il proprio traceback e con esso i frame del forward
    andato in OOM, quindi i tensori che l'hanno causato."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _degradation_levels(initial_batch, current_batch):
    """Scala di degrado [N, N//4, 1] derivata da INFERENCE_BATCH_SIZE, troncata
    ai livelli non superiori al batch corrente (il degrado e' permanente per il
    run, quindi non si risale mai).

    Cadere direttamente a 1 butterebbe via ~40x quando spesso basta un livello
    intermedio: entrambe le chiamate reali passano tutte le serie insieme ed e'
    TimesFM a spezzarle internamente.
    """
    levels = []
    for candidate in (initial_batch, max(1, initial_batch // 4), 1):
        if candidate <= current_batch and candidate not in levels:
            levels.append(candidate)
    return levels or [current_batch]


def _forecast_once(model, inputs, horizon, count_time):
    """Una chiamata a model.forecast, cronometrata.

    `list(inputs)`: TimesFM padda la lista in place fino a global_batch_size
    (timesfm_2p5_base.py). La prima chiamata resta corretta, ma un retry sulla
    stessa lista restituirebbe righe fantasma.

    A esito positivo marca `fl_has_produced_output`: e' quello che distingue un
    degrado che lascia il run uniforme da uno che lo spacca in due.
    """
    start = time.perf_counter()
    try:
        output, _ = model.forecast(horizon=horizon, inputs=list(inputs))
    finally:
        if count_time:
            model.fl_inference_seconds = (
                getattr(model, "fl_inference_seconds", 0.0) + time.perf_counter() - start
            )
    if count_time:
        model.fl_has_produced_output = True
    return output


def _to_array(value):
    """Normalizza un elemento di output (tensor torch o array numpy)."""
    return value.cpu().numpy() if hasattr(value, "cpu") else value


def _degrade_to(model, batch_size, count_time):
    """Ricompila a un batch size piu' basso e marca il run come degradato.

    `fl_degraded_after_inference` — il flag che dichiara il run NON consegnabile
    — si alza solo se almeno un'inferenza reale era gia' andata a buon fine a un
    batch diverso: e' quello il caso in cui i risultati del run sono stati
    calcolati a batch size diversi fra loro. Se il degrado avviene prima
    (nello smoke test, `count_time=False`, oppure al primo tentativo della prima
    chiamata reale, che non ha ancora prodotto nulla) il run gira uniformemente
    al batch degradato ed e' pienamente utilizzabile: segnalarlo costringerebbe
    a rifare per niente un run di minuti.
    """
    print(f"  Degrado del batch size a {batch_size} e nuovo tentativo...")
    model.fl_recompile(batch_size)
    model.fl_degraded = True
    if count_time and getattr(model, "fl_has_produced_output", False):
        model.fl_degraded_after_inference = True


def forecast_batch_with_fallback(model, inputs, horizon, count_time=True):
    """
    Forecast di una lista di serie, con degrado automatico in caso di problemi.

    Restituisce `(results, errors)`:
        results: list[np.ndarray | None] — una previsione per input, nello
                 stesso ordine, None per gli input falliti
        errors:  dict {indice_input: messaggio_errore}

    Tre livelli, nell'ordine:
      1. batch pieno al batch size corrente del modello;
      2. su OOM, degrado lungo la scala [N, N//4, 1] derivata da
         INFERENCE_BATCH_SIZE, un livello alla volta, via `fl_recompile`;
      3. su qualunque altro fallimento, degrado a batch 1 e poi un giro per
         singolo input (a batch 32 un loop per-input farebbe 32 serie di
         calcolo per ogni risultato utile, e con force_flip_invariance il
         decode gira due volte).

    Il degrado e' PERMANENTE per il run: evita di alternare avanti e indietro,
    ma non rende il run coerente. `run_backtest` chiama il modello una volta per
    origine e condivide l'istanza con il forecast futuro: se l'OOM scatta a meta'
    pipeline, quanto gia' calcolato resta al batch precedente. Un run con
    `fl_degraded_after_inference = True` va rifatto con INFERENCE_BATCH_SIZE piu'
    basso.

    Parametri:
        model:      istanza TimesFM da `setup_timesfm`
        inputs:     lista di serie (np.ndarray 1-D). Non viene mai mutata
        horizon:    mesi da prevedere
        count_time: True = il tempo si accumula in `model.fl_inference_seconds`.
                    False e' riservato allo smoke test, che a batch alto
                    inquinerebbe l'unica metrica di performance disponibile
    """
    results = [None] * len(inputs)
    errors = {}
    if not inputs:
        return results, errors

    current = getattr(model, "fl_batch_size", 1)
    initial = getattr(model, "fl_batch_size_initial", current)
    levels = _degradation_levels(initial, current)

    last_error = None
    for level in levels:
        if level != getattr(model, "fl_batch_size", level):
            _degrade_to(model, level, count_time)

        oom = False
        try:
            output = _forecast_once(model, inputs, horizon, count_time)
        except Exception as exc:
            last_error = exc
            oom = _is_oom(exc)
            if oom:
                print(f"  Memoria esaurita al batch {level}: {exc}")
            else:
                print(f"  Forecast in batch fallito ({exc}).")
        else:
            for i in range(len(inputs)):
                results[i] = _to_array(output[i])
            return results, errors

        # Fuori dall'except: solo qui i frame del forward andato in OOM sono
        # stati rilasciati e empty_cache() ha davvero effetto.
        _free_cuda_memory()
        if not oom:
            break

    # Livello 3: batch 1 e poi un input alla volta.
    if getattr(model, "fl_batch_size", 1) != 1:
        _degrade_to(model, 1, count_time)
    print(f"  Fallback a forecast per singola serie ({len(inputs)} serie)...")

    for i, series in enumerate(inputs):
        error = None
        try:
            output = _forecast_once(model, [series], horizon, count_time)
            results[i] = _to_array(output[0])
        except Exception as exc:
            error = str(exc)
        if error is not None:
            # Fuori dall'except, per lo stesso motivo del ciclo qui sopra: dentro,
            # l'eccezione tiene vivo il traceback e con esso i tensori che hanno
            # esaurito la memoria, e empty_cache() non libererebbe nulla — proprio
            # mentre stiamo per chiedere al modello la serie successiva.
            errors[i] = error
            _free_cuda_memory()

    if errors and last_error is not None:
        print(f"  {len(errors)} serie non previste. Primo errore in batch: {last_error}")

    return results, errors


def forecast_all_skus_point(model, series_dict, horizon, verbose=False):
    """
    Forecast point (mediana) per tutte le serie SKU in un'unica chiamata batch,
    con il degrado automatico di `forecast_batch_with_fallback`.

    Parametri:
        model:        istanza TimesFM (da `setup_timesfm`)
        series_dict:  dict {SKU: lista_valori_storici}
        horizon:      mesi da prevedere
        verbose:      se True, elenca gli SKU falliti

    Restituisce:
        results: dict {SKU: array_forecast}
        errors:  dict {SKU: messaggio_errore} per gli SKU falliti
    """
    results = {}
    errors = {}

    total = len(series_dict)
    print(f"Avvio forecast point per {total} SKU (batch)...")

    skus = list(series_dict.keys())
    inputs = [np.array(series_dict[s], dtype=np.float32) for s in skus]

    preds, batch_errors = forecast_batch_with_fallback(model, inputs, horizon)

    for i, sku in enumerate(skus):
        if preds[i] is not None:
            results[sku] = preds[i]
        else:
            errors[sku] = batch_errors.get(i, "forecast non disponibile")
            if verbose:
                print(f"  SKU {sku}: {errors[sku]}")

    print("Forecast point completato.")
    print(" - SKU riusciti:", len(results))
    print(" - SKU falliti:", len(errors))

    return results, errors

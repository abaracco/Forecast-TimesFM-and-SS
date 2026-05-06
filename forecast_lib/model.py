"""
Modulo F — Caricamento modello TimesFM e funzione di forecast batch.

Il loader fa diverse cose, in ordine:
  1. Imposta la cache HuggingFace (effimera in Colab, persistente in locale)
  2. Clona il repository ufficiale TimesFM se non presente
  3. Aggiunge la cartella sorgente a sys.path
  4. Importa manualmente il modulo PyTorch tramite spec_from_file_location
     (evita conflitti col pacchetto pip "timesfm", che non sempre e'
     compatibile con Python 3.12 di Colab)
  5. Identifica la classe modello (quella con metodo `forecast`)
  6. Scarica/aggiorna i pesi pre-addestrati da HuggingFace
  7. Configura ForecastConfig (max_context, max_horizon, etc.)
  8. Sposta il modello su GPU se disponibile, altrimenti CPU
  9. Esegue uno smoke test rapido per verificare che il modello risponda

`forecast_all_skus_point` lancia il forecast in batch su tutti gli SKU,
con fallback automatico a chiamate per-SKU se il batch fallisce.
"""

import importlib
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np


def setup_timesfm(
    *,
    colab,
    horizon,
    model_id="google/timesfm-2.5-200m-pytorch",
    timesfm_repo_url="https://github.com/google-research/timesfm.git",
):
    """
    Carica il modello TimesFM gestendo Colab e locale in modo trasparente.

    Restituisce l'istanza del modello compilato e pronta all'inferenza.

    Parametri:
        colab:             True per modalita' Colab, False per locale
        horizon:           orizzonte massimo di forecast (passato a ForecastConfig)
        model_id:          nome del modello su HuggingFace
        timesfm_repo_url:  URL del repository TimesFM da clonare
    """
    # 1. Cache HuggingFace
    if colab:
        os.environ.setdefault("HF_HOME", "/content/.cache/huggingface")
    # In locale: HF_HOME non viene impostato, usa il default (~/.cache/huggingface).
    # Il check ETag automatico aggiorna il modello solo se necessario.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # 2. Clone del repository TimesFM (solo codice sorgente)
    if colab:
        timesfm_dir = "/content/timesfm"
    else:
        timesfm_dir = "./timesfm"

    if not pathlib.Path(timesfm_dir).exists():
        print("Clone del repository TimesFM in corso...")
        subprocess.run(
            ["git", "clone", "-q", timesfm_repo_url, timesfm_dir],
            check=True,
        )

    # 3. Aggiungi le cartelle sorgente a sys.path
    pkg_dir = os.path.join(timesfm_dir, "src")
    t25_dir = os.path.join(timesfm_dir, "src", "timesfm", "timesfm_2p5")

    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    # 4. Import manuale del modulo PyTorch
    torch_files = list(pathlib.Path(t25_dir).glob("**/*pytorch*.py"))
    if not torch_files:
        torch_files = list(pathlib.Path(t25_dir).glob("**/*torch*.py"))
    torch_files.sort()

    if not torch_files:
        raise RuntimeError("Non trovo il modulo Torch in timesfm_2p5!")

    torch_mod_path = torch_files[0]
    print("Modulo trovato:", torch_mod_path)

    spec = importlib.util.spec_from_file_location(
        "timesfm.timesfm_2p5.timesfm_2p5_torch",
        str(torch_mod_path),
    )
    torch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch_mod)
    sys.modules["timesfm.timesfm_2p5.timesfm_2p5_torch"] = torch_mod

    # 5. Identifica la classe modello (quella con metodo "forecast")
    ModelClass = None
    for name, cls in vars(torch_mod).items():
        if isinstance(cls, type) and hasattr(cls, "forecast"):
            ModelClass = cls
            break

    if ModelClass is None:
        raise RuntimeError("Nessuna classe TimesFM con metodo forecast trovata!")

    print("Classe modello:", ModelClass)

    # 6. Scarica i pesi pre-addestrati
    print("Download pesi (HuggingFace)...")
    model = ModelClass.from_pretrained(model_id)

    # 7. Configurazione parametri di inferenza
    cfg = None
    for mp in [
        "timesfm.config",
        "timesfm.configs",
        "timesfm.timesfm_2p5.configs.forecast_config",
    ]:
        try:
            m = importlib.import_module(mp)
            if hasattr(m, "ForecastConfig"):
                ForecastConfig = m.ForecastConfig
                cfg = ForecastConfig(
                    max_context=512,
                    max_horizon=horizon,
                    normalize_inputs=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
                break
        except Exception:
            pass

    if cfg is not None:
        model.compile(cfg)
    else:
        model.compile()

    # 8. Rilevamento automatico CPU/GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    print("Modello pronto (loader manuale TimesFM attivato).")

    # 9. Smoke test
    try:
        test = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19], dtype=np.float32)
        pred, _ = model.forecast(horizon=3, inputs=[test])
        print("Smoke test OK ->", pred[0])
    except Exception as e:
        print("Smoke test fallito:", e)

    return model


def forecast_all_skus_point(model, series_dict, horizon, verbose=False):
    """
    Forecast point (mediana) per tutte le serie SKU in un'unica chiamata batch.
    Se il batch fallisce, fallback automatico a chiamate per-SKU.

    Parametri:
        model:        istanza TimesFM (da `setup_timesfm`)
        series_dict:  dict {SKU: lista_valori_storici}
        horizon:      mesi da prevedere
        verbose:      se True, log per ogni SKU nel fallback per-SKU

    Restituisce:
        results: dict {SKU: array_forecast}
        errors:  dict {SKU: messaggio_errore} per gli SKU falliti nel fallback
    """
    results = {}
    errors = {}

    total = len(series_dict)
    print(f"Avvio forecast point per {total} SKU (batch)...")

    skus = list(series_dict.keys())
    inputs = [np.array(series_dict[s], dtype=np.float32) for s in skus]

    try:
        # Tentativo batch: tutte le serie in una sola chiamata
        all_fc, _ = model.forecast(horizon=horizon, inputs=inputs)

        for i, sku in enumerate(skus):
            fc = all_fc[i]
            if hasattr(fc, "cpu"):
                fc = fc.cpu().numpy()
            results[sku] = fc

    except Exception as e:
        # Fallback: forecast singolo per ogni SKU
        print(f"Batch fallito ({e}), fallback a forecast singolo...")
        for idx, sku in enumerate(skus):
            if verbose:
                print(f"[{idx+1}/{total}] SKU {sku}... ", end="")
            try:
                point_fc, _ = model.forecast(
                    horizon=horizon,
                    inputs=[inputs[idx]],
                )
                fc = point_fc[0]
                if hasattr(fc, "cpu"):
                    fc = fc.cpu().numpy()
                results[sku] = fc
                if verbose:
                    print("OK")
            except Exception as e2:
                if verbose:
                    print("Errore")
                errors[sku] = str(e2)

    print("Forecast point completato.")
    print(" - SKU riusciti:", len(results))
    print(" - SKU falliti:", len(errors))

    return results, errors

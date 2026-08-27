"""
Modulo J — Costruzione tabella finale ed export Excel.

La parte di I/O specifica per ambiente (download Colab vs save dialog
locale) resta nel notebook. Qui vive solo la logica di:
  1. Pivot dello storico in formato wide
  2. Rinomina colonne forecast (prefisso 'f' per distinguerle dallo storico)
  3. Merge: metadati + inventario + storico + forecast
  4. Salvataggio Excel (+ foglio "Run info" opzionale)
  5. CSV di audit per ricostruire a posteriori un run

Vincolo sull'ordine dei fogli: la tabella dati resta SEMPRE il primo foglio.
Questo stesso progetto legge il file di input con `list(all_sheets.keys())[0]`,
e non e' l'unico consumatore possibile.
"""

import os
import re

import pandas as pd

DATA_SHEET_NAME = "Sheet1"      # nome che pandas assegna di default: non cambiarlo
RUN_INFO_SHEET_NAME = "Run info"


def build_forecast_wide(df_fc_long, id_col):
    """
    Converte il forecast da long a wide ordinando le colonne mese
    cronologicamente.

    Parametri:
        df_fc_long: DataFrame con colonne [id_col, "Period", "Forecast"]
        id_col:     nome colonna SKU
    """
    df_fc_wide = df_fc_long.pivot(
        index=id_col,
        columns="Period",
        values="Forecast"
    ).reset_index()

    forecast_cols = sorted(
        [c for c in df_fc_wide.columns if re.match(r"^\d{4}_\d{2}$", str(c))]
    )
    return df_fc_wide[[id_col] + forecast_cols]


def build_history_wide(df_filtered, id_col):
    """
    Converte lo storico (long) in formato wide pivotando su 'Period'.
    """
    return df_filtered.pivot(
        index=id_col,
        columns="Period",
        values="Demand"
    ).reset_index()


def build_final_table(
    df_filtered,
    df_fc_wide,
    df_inventory,
    *,
    id_col,
    desc_col,
    pack_size_col,
    uom_col,
):
    """
    Merge finale: metadati base + inventario (opzionale) + storico + forecast.

    Le colonne forecast vengono prefissate con 'f' per distinguerle dalle
    colonne storiche con lo stesso pattern 'YYYY_MM'.

    Parametri:
        df_filtered:    storico in formato long (post winsorize)
        df_fc_wide:     forecast in formato wide
        df_inventory:   risultato di calculate_inventory_logic, oppure None se disabilitato
        id_col, desc_col, pack_size_col, uom_col: nomi colonne (dal Modulo A)
    """
    # 1. Metadati base
    meta = df_filtered[[id_col, desc_col, pack_size_col, uom_col]].drop_duplicates()

    # 2. Inventario (se calcolato)
    if df_inventory is not None:
        df_inv_clean = df_inventory.rename(columns={"LT_Final": "LT"})
        out_step1 = meta.merge(df_inv_clean, on=id_col, how="left")
    else:
        out_step1 = meta.copy()

    # 3. Storico in formato wide
    df_hist_wide = build_history_wide(df_filtered, id_col)

    # 4. Forecast in formato wide con prefisso 'f'
    df_fc_pref = df_fc_wide.copy()
    df_fc_pref.rename(
        columns={c: f"f{c}" for c in df_fc_pref.columns if c != id_col},
        inplace=True
    )

    # 5. Merge finale
    out_final = out_step1.merge(df_hist_wide, on=id_col, how="left")
    out_final = out_final.merge(df_fc_pref, on=id_col, how="left")

    # SKU come stringa per evitare problemi di formattazione in Excel
    out_final[id_col] = out_final[id_col].astype(str)

    # NaN nella safety stock -> 0
    if "SafetyStock" in out_final.columns:
        out_final["SafetyStock"] = out_final["SafetyStock"].fillna(0)

    return out_final


def save_excel(df, path, run_info=None):
    """Salva il DataFrame in formato Excel (indice escluso).

    Con `run_info` (dict campo -> valore) aggiunge un SECONDO foglio "Run info".
    La tabella dati resta il primo foglio in entrambi i casi.
    """
    if run_info is None:
        df.to_excel(path, index=False)
        return

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name=DATA_SHEET_NAME, index=False)
        run_info_to_frame(run_info).to_excel(
            writer, sheet_name=RUN_INFO_SHEET_NAME, index=False
        )


def run_info_to_frame(run_info):
    """dict ordinato -> DataFrame a due colonne, pronto per il foglio."""
    return pd.DataFrame(
        {"Campo": list(run_info.keys()),
         "Valore": [_as_cell(v) for v in run_info.values()]}
    )


def _as_cell(value):
    """Valori non scrivibili come cella Excel -> stringa; None -> vuoto."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "si" if value else "no"
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def build_run_info(
    *,
    model=None,
    timestamp=None,
    lib_version=None,
    timesfm_version=None,
    model_id=None,
    model_revision=None,
    inference_batch_size=None,
    business_adjustment_factor=None,
    run_backtest=None,
    n_backtest_origins=None,
    shrinkage_enabled=None,
    rounding_mode=None,
    q_global=None,
    kpi_motul=None,
    n_backtest_skus=None,
    n_skus_excluded=None,
    n_skus_zero_accuracy=None,
    n_forecast_errors=None,
):
    """Raccoglie i campi del foglio "Run info": tutto cio' che serve a
    ricostruire a posteriori come e' stato prodotto un file di output.

    Lo stato del modello viene letto dagli attributi `fl_*` attaccati da
    `setup_timesfm`, quindi il chiamante deve passare solo i parametri del
    Modulo A e i numeri calcolati dalla pipeline. `model=None` e' ammesso
    (utile nei test): i campi corrispondenti restano vuoti.

    `model_revision` (il valore chiesto) e `fl_model_revision` (quello risolto)
    sono due campi distinti di proposito: il secondo puo' mancare senza che il
    run sia sbagliato, il primo no.
    """
    fl = (lambda name, default=None: getattr(model, name, default))

    return {
        "Data e ora del run": timestamp,
        "forecast_lib": lib_version,
        "TimesFM (tag pinnato)": fl("fl_timesfm_tag", timesfm_version),
        "Pin TimesFM verificato": fl("fl_pin_verified"),
        "Modello HuggingFace": model_id,
        # La revision RICHIESTA va registrata anche quando quella risolta manca:
        # `_resolve_model_revision` e' diagnostica e non bloccante (torna None se
        # si e' offline o se cambia il layout della cache di HuggingFace), e senza
        # questo campo il file non conserverebbe alcuna traccia di quali pesi
        # erano stati chiesti — cioe' proprio il dato su cui poggia la
        # riproducibilita' del run.
        "Revision pesi richiesta": model_revision,
        "Revision pesi": fl("fl_model_revision"),
        "Device": fl("fl_device"),
        "INFERENCE_BATCH_SIZE": inference_batch_size,
        "Batch size effettivo": fl("fl_batch_size"),
        "global_batch_size": fl("global_batch_size"),
        "Batch degradato durante il run": fl("fl_degraded"),
        "Degrado dopo inferenza reale": fl("fl_degraded_after_inference"),
        "Tempo di inferenza (s)": fl("fl_inference_seconds"),
        "RUN_BACKTEST": run_backtest,
        "N_BACKTEST_ORIGINS": n_backtest_origins,
        "SHRINKAGE_ENABLED": shrinkage_enabled,
        "q globale (mediana shrinkage)": q_global,
        "KPI Motul pesato": kpi_motul,
        "SKU con risultato di backtest": n_backtest_skus,
        "SKU esclusi dal backtest": n_skus_excluded,
        "SKU con accuratezza nulla": n_skus_zero_accuracy,
        "SKU senza forecast": n_forecast_errors,
        "BUSINESS_ADJUSTMENT_FACTOR": business_adjustment_factor,
        "ROUNDING_MODE": rounding_mode,
    }


def save_audit_csvs(df_backtest_results, fc_errors, output_dir, file_base, suffix):
    """Scrive i due CSV di audit accanto all'output Excel e ne restituisce i path.

    Naming coerente con il file Excel: `suffix` inizia gia' con uno spazio.
    `q_global` vive in `df.attrs`, che non sopravvive a `to_csv`: viene
    materializzato come colonna costante.
    """
    paths = []

    df_bt = df_backtest_results.copy()
    if "q_global" not in df_bt.columns:
        df_bt["q_global"] = df_backtest_results.attrs.get("q_global")
    bt_path = os.path.join(output_dir, f"{file_base} backtest{suffix}.csv")
    df_bt.to_csv(bt_path, index=False)
    paths.append(bt_path)

    df_err = pd.DataFrame(
        {"SKU": list(fc_errors.keys()), "Errore": list(fc_errors.values())}
    )
    err_path = os.path.join(output_dir, f"{file_base} errori{suffix}.csv")
    df_err.to_csv(err_path, index=False)
    paths.append(err_path)

    return paths

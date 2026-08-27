"""
Confronto fra due run del forecast: gate G1-G5 e diagnostica (§ 9.2 del piano).

Non e' un test pytest: e' lo strumento con cui si valutano i collaudi manuali
T4.1, T4.2 e T4b. Legge gli artefatti di due run (il file Excel di output e i due
CSV di audit), calcola i gate e stampa un report in Markdown.

    python tests/tools/compare_forecast_outputs.py \\
        --baseline "output/Run A.xlsx" --new "output/Run B.xlsx" \\
        --baseline-backtest "output/Run A backtest.csv" \\
        --new-backtest "output/Run B backtest.csv" \\
        --mode refactor

I tre modi corrispondono ai tre collaudi:

    refactor   T4.1  run A (main) vs run B (nuovo, batch 1)      -> G1
    batch      T4.2  run B vs run C (batch 32)                   -> G2 G3 G4 G5
    nobacktest T4b   run B' vs run C' (RUN_BACKTEST = False)     -> G2 G3 G4
                     con soglie G3/G4 a 0.05%: senza backtest non ci sono flip
                     di q, restano solo le differenze di arrotondamento

Codice di uscita: 0 se tutti i gate applicabili passano, 1 altrimenti.

Due tolleranze volute:
  - il foglio "Run info" e' ESCLUSO dal confronto (contiene data/ora e nella run
    A non esiste affatto) ma i suoi campi finiscono nel report;
  - i due CSV di backtest possono avere insiemi di colonne diversi (la run A non
    ha `BestQuantileRaw`, `BestAccuracyRaw`, `q_global`, arrivati con la Fase
    1.5): si confronta l'intersezione e si dichiara cosa e' stato ignorato.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


FORECAST_COL_RE = re.compile(r"^f\d{4}_\d{2}$")
RUN_INFO_SHEET_NAME = "Run info"

# Colonne che devono coincidere per costruzione: non dipendono dal forecast.
STRUCTURAL_COLS = ["ABC", "XYZ", "SafetyStock", "LT"]

# Soglie di § 9.2. Quelle di T4b sono piu' strette di un ordine di grandezza:
# senza backtest non ci sono flip di q, quindi resta solo l'arrotondamento.
DEFAULT_THRESHOLDS = {
    "batch": {"g3": 0.005, "g4": 0.01},
    "nobacktest": {"g3": 0.0005, "g4": 0.0005},
    "refactor": {"g3": 0.0, "g4": 0.0},
}
G5_SKU_THRESHOLD = 0.05         # |Δ volume| oltre cui serve una spiegazione
AB_REPORT_THRESHOLD = 0.04      # elenco completo degli SKU A/B oltre questa soglia
KPI_ALERT_PP = 0.5              # peggioramento del KPI che impone un'indagine


# ======================================================================
# Caricamento degli artefatti
# ======================================================================

class Run:
    """Gli artefatti di una singola run, gia' indicizzati per SKU."""

    def __init__(self, label, data, run_info, backtest, errors, id_col):
        self.label = label
        self.data = data
        self.run_info = run_info
        self.backtest = backtest
        self.errors = errors
        self.id_col = id_col

    @property
    def forecast_cols(self):
        return [c for c in self.data.columns if FORECAST_COL_RE.match(str(c))]

    @property
    def volume(self):
        """Volume previsto per SKU: somma dei mesi di forecast (NaN = 0)."""
        cols = self.forecast_cols
        if not cols:
            return pd.Series(0.0, index=self.data.index, dtype=float)
        return self.data[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    @property
    def backtest_skus(self):
        if self.backtest is None:
            return None
        return set(self.backtest.index)

    @property
    def error_skus(self):
        if self.errors is None:
            return None
        return set(self.errors)

    @property
    def q_global(self):
        if self.backtest is not None and "q_global" in self.backtest.columns:
            values = self.backtest["q_global"].dropna().unique()
            if len(values):
                return float(values[0])
        value = self.run_info.get("q globale (mediana shrinkage)")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def load_run(label, xlsx_path, backtest_csv=None, errors_csv=None, id_col="SKU"):
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    if not sheets:
        raise SystemExit(f"{xlsx_path}: nessun foglio nel file.")

    # La tabella dati e' SEMPRE il primo foglio (vincolo di export.save_excel).
    data_sheet = list(sheets.keys())[0]
    if data_sheet == RUN_INFO_SHEET_NAME:
        raise SystemExit(
            f"{xlsx_path}: il primo foglio e' \"{RUN_INFO_SHEET_NAME}\". "
            f"Il file non rispetta il vincolo sull'ordine dei fogli."
        )
    data = sheets[data_sheet]
    if id_col not in data.columns:
        raise SystemExit(f"{xlsx_path}: colonna '{id_col}' assente dal primo foglio.")
    data = data.copy()
    data[id_col] = data[id_col].astype(str)
    data = data.set_index(id_col).sort_index()

    run_info = {}
    if RUN_INFO_SHEET_NAME in sheets:
        sheet = sheets[RUN_INFO_SHEET_NAME]
        if {"Campo", "Valore"}.issubset(sheet.columns):
            run_info = dict(zip(sheet["Campo"], sheet["Valore"]))

    backtest = None
    if backtest_csv:
        backtest = pd.read_csv(backtest_csv)
        if id_col not in backtest.columns:
            raise SystemExit(f"{backtest_csv}: colonna '{id_col}' assente.")
        backtest[id_col] = backtest[id_col].astype(str)
        backtest = backtest.set_index(id_col).sort_index()

    errors = None
    if errors_csv:
        err = pd.read_csv(errors_csv)
        col = id_col if id_col in err.columns else err.columns[0]
        errors = set(err[col].astype(str))

    return Run(label, data, run_info, backtest, errors, id_col)


# ======================================================================
# Gate
# ======================================================================

class Gate:
    def __init__(self, name, title, passed, details, applicable=True):
        self.name = name
        self.title = title
        self.passed = passed
        self.details = details
        self.applicable = applicable

    @property
    def verdict(self):
        if not self.applicable:
            return "n/a"
        return "PASS" if self.passed else "FAIL"


def _frames_equal(a, b):
    """Confronto NaN-aware fra due DataFrame gia' allineati."""
    return a.equals(b)


def gate_g1(base, new):
    """G1 — identita' del refactor: colonne forecast e CSV di backtest
    bit-identici. Una differenza qui e' un bug, non un effetto del batching."""
    details = []
    ok = True

    if list(base.data.index) != list(new.data.index):
        ok = False
        only_base = sorted(set(base.data.index) - set(new.data.index))
        only_new = sorted(set(new.data.index) - set(base.data.index))
        details.append(f"insiemi di SKU diversi: solo baseline {only_base[:10]}, "
                       f"solo nuovo {only_new[:10]}")
        common = base.data.index.intersection(new.data.index)
    else:
        common = base.data.index

    cols_base, cols_new = set(base.forecast_cols), set(new.forecast_cols)
    if cols_base != cols_new:
        ok = False
        details.append(f"colonne forecast diverse: solo baseline "
                       f"{sorted(cols_base - cols_new)}, solo nuovo "
                       f"{sorted(cols_new - cols_base)}")
    cols = sorted(cols_base & cols_new)

    if cols:
        a = base.data.loc[common, cols]
        b = new.data.loc[common, cols]
        if _frames_equal(a, b):
            details.append(f"forecast bit-identico su {len(cols)} colonne "
                           f"x {len(common)} SKU (confronto NaN-aware)")
        else:
            ok = False
            diff = _first_differences(a, b, limit=10)
            details.append(f"{len(diff)} differenze mostrate (prime 10):")
            details.extend(f"  {d}" for d in diff)
    else:
        ok = False
        details.append("nessuna colonna forecast in comune")

    if base.backtest is not None and new.backtest is not None:
        details.extend(_compare_backtest_csv(base, new))
        if any(d.startswith("DIFFERENZE") for d in details):
            ok = False

    return Gate("G1", "Identita' del refactor", ok, details)


def _first_differences(a, b, limit=10):
    """Prime differenze cella per cella fra due frame allineati."""
    out = []
    for col in a.columns:
        left = pd.to_numeric(a[col], errors="coerce")
        right = pd.to_numeric(b[col], errors="coerce")
        mismatch = ~((left == right) | (left.isna() & right.isna()))
        for sku in a.index[mismatch]:
            out.append(f"{sku} / {col}: {a.at[sku, col]!r} -> {b.at[sku, col]!r}")
            if len(out) >= limit:
                return out
    return out


def _compare_backtest_csv(base, new):
    details = []
    common_cols = sorted(set(base.backtest.columns) & set(new.backtest.columns))
    ignored = sorted((set(base.backtest.columns) ^ set(new.backtest.columns)))
    if ignored:
        details.append(f"colonne di backtest presenti in una sola run, ignorate: "
                       f"{ignored}")
    if set(base.backtest.index) != set(new.backtest.index):
        details.append("DIFFERENZE: l'insieme degli SKU nel CSV di backtest e' diverso")
        return details

    idx = base.backtest.index
    a = base.backtest.loc[idx, common_cols]
    b = new.backtest.loc[idx, common_cols]
    if _frames_equal(a, b):
        details.append(f"CSV di backtest identico su {len(common_cols)} colonne "
                       f"x {len(idx)} SKU")
    else:
        details.append("DIFFERENZE nel CSV di backtest (prime 10):")
        details.extend(f"  {d}" for d in _first_differences(a, b, limit=10))
    return details


def gate_g2(base, new):
    """G2 — identita' strutturale: ABC/XYZ/SafetyStock/LT e gli insiemi di SKU
    con risultato di backtest e falliti."""
    details = []
    ok = True

    common = base.data.index.intersection(new.data.index)
    if len(common) != len(base.data.index) or len(common) != len(new.data.index):
        ok = False
        details.append(f"insiemi di SKU diversi: baseline {len(base.data.index)}, "
                       f"nuovo {len(new.data.index)}, in comune {len(common)}")

    for col in STRUCTURAL_COLS:
        if col not in base.data.columns or col not in new.data.columns:
            details.append(f"{col}: assente in almeno una run, non confrontata")
            continue
        a = base.data.loc[common, [col]]
        b = new.data.loc[common, [col]]
        if _frames_equal(a, b):
            details.append(f"{col}: identica")
        else:
            ok = False
            diffs = _first_differences(a, b, limit=5)
            details.append(f"{col}: DIVERSA ({len(diffs)} differenze mostrate)")
            details.extend(f"  {d}" for d in diffs)

    if base.backtest_skus is not None and new.backtest_skus is not None:
        if base.backtest_skus == new.backtest_skus:
            details.append(f"SKU con risultato di backtest: identici "
                           f"({len(base.backtest_skus)})")
        else:
            ok = False
            details.append(
                f"SKU con risultato di backtest DIVERSI: solo baseline "
                f"{sorted(base.backtest_skus - new.backtest_skus)[:10]}, solo nuovo "
                f"{sorted(new.backtest_skus - base.backtest_skus)[:10]} "
                f"(chi sparisce ricade su q = 0.5 nel Modulo H: fino a 5x)"
            )
    else:
        details.append("CSV di backtest non forniti: insieme degli SKU non confrontato")

    if base.error_skus is not None and new.error_skus is not None:
        if base.error_skus == new.error_skus:
            details.append(f"SKU senza forecast: identici ({len(base.error_skus)})")
        else:
            ok = False
            details.append(
                f"SKU senza forecast DIVERSI: solo baseline "
                f"{sorted(base.error_skus - new.error_skus)[:10]}, solo nuovo "
                f"{sorted(new.error_skus - base.error_skus)[:10]}"
            )
    else:
        details.append("CSV errori non forniti: insieme degli SKU falliti non confrontato")

    return Gate("G2", "Identita' strutturale", ok, details)


def _volumes(base, new):
    common = base.data.index.intersection(new.data.index)
    return base.volume.reindex(common).fillna(0.0), new.volume.reindex(common).fillna(0.0)


def gate_g3(base, new, threshold):
    """G3 — impatto aggregato CON segno."""
    vb, vn = _volumes(base, new)
    total_b, total_n = float(vb.sum()), float(vn.sum())
    delta = (total_n - total_b) / total_b if total_b else (0.0 if total_n == 0 else np.inf)
    ok = abs(delta) <= threshold
    details = [
        f"volume baseline {total_b:,.1f} -> nuovo {total_n:,.1f}",
        f"scostamento {delta:+.4%} (soglia {threshold:.4%})",
    ]
    return Gate("G3", "Impatto aggregato con segno", ok, details)


def gate_g4(base, new, threshold):
    """G4 — impatto aggregato NON compensativo: G3 ha segno e si compensa."""
    vb, vn = _volumes(base, new)
    total_b = float(vb.sum())
    abs_delta = float((vn - vb).abs().sum())
    ratio = abs_delta / total_b if total_b else (0.0 if abs_delta == 0 else np.inf)
    ok = ratio <= threshold
    details = [
        f"somma degli scostamenti assoluti {abs_delta:,.1f} su volume "
        f"baseline {total_b:,.1f}",
        f"rapporto {ratio:.4%} (soglia {threshold:.4%})",
    ]
    return Gate("G4", "Impatto aggregato non compensativo", ok, details)


def _relative_delta(base, new):
    """Scostamento relativo per SKU. Volume baseline nullo e nuovo non nullo ->
    infinito: e' una differenza inspiegabile quanto le altre, non una divisione
    per zero da nascondere."""
    vb, vn = _volumes(base, new)
    delta = vn - vb
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = delta.abs() / vb.replace(0.0, np.nan)
    rel = rel.where(~(vb == 0), other=np.where(delta.abs() > 0, np.inf, 0.0))
    return vb, vn, delta, rel


def _quantile_map(run, column="BestQuantile"):
    if run.backtest is None or column not in run.backtest.columns:
        return None
    return run.backtest[column]


def gate_g5(base, new):
    """G5 — spiegabilita': ogni SKU con |Δ volume| > 5% deve avere un
    `BestQuantile` diverso fra le due run. E' il gate discriminante."""
    qb = _quantile_map(base)
    qn = _quantile_map(new)
    if qb is None or qn is None:
        return Gate("G5", "Spiegabilita' degli scostamenti", True,
                    ["BestQuantile non disponibile in almeno una run: gate non valutato"],
                    applicable=False)

    _vb, _vn, delta, rel = _relative_delta(base, new)
    suspects = rel[rel > G5_SKU_THRESHOLD].sort_values(ascending=False)

    unexplained = []
    for sku in suspects.index:
        q_before = qb.get(sku, None)
        q_after = qn.get(sku, None)
        same = (
            (q_before is None and q_after is None)
            or (q_before is not None and q_after is not None
                and float(q_before) == float(q_after))
        )
        if same:
            unexplained.append((sku, rel[sku], delta[sku], q_before, q_after))

    details = [
        f"SKU con |Δ volume| > {G5_SKU_THRESHOLD:.0%}: {len(suspects)}",
        f"di cui senza un cambio di BestQuantile: {len(unexplained)}",
    ]
    for sku, r, d, q_before, q_after in unexplained[:20]:
        details.append(f"  {sku}: Δ {d:+,.1f} ({r:.1%}), q {q_before} -> {q_after}")
    if not unexplained:
        details.append("ogni scostamento grande e' accompagnato da un cambio di q")

    return Gate("G5", "Spiegabilita' degli scostamenti", not unexplained, details)


# ======================================================================
# Diagnostica obbligatoria (§ 9.2) — riportata, non gate
# ======================================================================

def _kpi_motul(run, skus):
    if run.backtest is None:
        return None
    cols = run.backtest.columns
    if "BestAccuracy" not in cols or "TotalWeight" not in cols:
        return None
    sub = run.backtest.reindex(skus).dropna(subset=["BestAccuracy", "TotalWeight"])
    weight = sub["TotalWeight"].sum()
    if not weight:
        return None
    return float((sub["BestAccuracy"] * sub["TotalWeight"]).sum() / weight)


def _changed_quantiles(base, new, column):
    qb = _quantile_map(base, column)
    qn = _quantile_map(new, column)
    if qb is None or qn is None:
        return None
    common = qb.index.intersection(qn.index)
    if not len(common):
        return None
    changed = (qb.loc[common].astype(float) != qn.loc[common].astype(float)).sum()
    return int(changed), len(common)


def diagnostics(base, new):
    lines = []

    for column in ("BestQuantileRaw", "BestQuantile"):
        result = _changed_quantiles(base, new, column)
        if result is None:
            lines.append(f"- `{column}`: non disponibile in entrambe le run")
        else:
            changed, total = result
            share = changed / total if total else 0.0
            lines.append(f"- SKU con `{column}` diverso: **{changed}** su {total} "
                         f"({share:.2%})")

    for run in (base, new):
        if run.backtest is not None and "BestAccuracyRaw" in run.backtest.columns:
            zero = int((run.backtest["BestAccuracyRaw"] <= 0).sum())
            lines.append(f"- {run.label}: SKU con `BestAccuracyRaw == 0` "
                         f"(q arbitrario per costruzione): **{zero}**")

    lines.append(f"- `q_global`: {base.q_global} -> {new.q_global}")

    if base.backtest is not None and new.backtest is not None:
        common = base.backtest.index.intersection(new.backtest.index)
        kpi_b = _kpi_motul(base, common)
        kpi_n = _kpi_motul(new, common)
        if kpi_b is not None and kpi_n is not None:
            delta_pp = (kpi_n - kpi_b) * 100
            flag = ""
            if delta_pp < -KPI_ALERT_PP:
                flag = (f"  <- PEGGIORAMENTO oltre {KPI_ALERT_PP} pp: indagare "
                        f"prima di proseguire (§ 9.3)")
            lines.append(f"- KPI Motul pesato (intersezione, {len(common)} SKU): "
                         f"{kpi_b:.4%} -> {kpi_n:.4%} ({delta_pp:+.4f} pp)"
                         f"{flag}")
            lines.append("  *sanity check, non gate: `BestAccuracy` e' "
                         "auto-selezionato.*")

    return lines


def _top_table(rows, headers):
    if not rows:
        return ["(nessuno)"]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return out


def deviation_tables(base, new, limit=20):
    """Top-20 per scostamento assoluto E per scostamento relativo.

    Il secondo e' obbligatorio: senza, la classe C — la maggioranza degli SKU —
    non comparirebbe mai in alcun output del collaudo, e uno SKU C con q che
    salta da 0.50 a 0.10 (rottura di stock su un prodotto vero) resterebbe
    invisibile.
    """
    vb, vn, delta, rel = _relative_delta(base, new)
    qb = _quantile_map(base)
    qn = _quantile_map(new)
    abc = base.data["ABC"] if "ABC" in base.data.columns else None

    def row(sku):
        return [
            sku,
            "" if abc is None else abc.get(sku, ""),
            f"{vb[sku]:,.1f}",
            f"{vn[sku]:,.1f}",
            f"{delta[sku]:+,.1f}",
            "inf" if np.isinf(rel[sku]) else f"{rel[sku]:.2%}",
            "n/d" if qb is None else qb.get(sku, "n/d"),
            "n/d" if qn is None else qn.get(sku, "n/d"),
        ]

    headers = ["SKU", "ABC", "volume base", "volume nuovo", "Δ", "Δ %", "q base", "q nuovo"]

    by_abs = delta.abs().sort_values(ascending=False).head(limit).index
    by_rel = rel.sort_values(ascending=False).head(limit).index

    sections = []
    sections.append(("Top-20 per scostamento assoluto",
                     _top_table([row(s) for s in by_abs if delta[s] != 0], headers)))
    sections.append(("Top-20 per scostamento relativo",
                     _top_table([row(s) for s in by_rel if delta[s] != 0], headers)))

    if abc is not None:
        ab_mask = abc.reindex(rel.index).isin(["A", "B"]) & (rel > AB_REPORT_THRESHOLD)
        ab_skus = rel[ab_mask.fillna(False)].sort_values(ascending=False).index
        sections.append((
            f"SKU di classe A o B con |Δ volume| > {AB_REPORT_THRESHOLD:.0%} "
            f"(elenco completo)",
            _top_table([row(s) for s in ab_skus], headers),
        ))

    return sections


# ======================================================================
# Report
# ======================================================================

RUN_INFO_FIELDS = [
    "Data e ora del run", "forecast_lib", "TimesFM (tag pinnato)",
    "Pin TimesFM verificato", "Revision pesi", "Device", "INFERENCE_BATCH_SIZE",
    "Batch size effettivo", "Batch degradato durante il run",
    "Degrado dopo inferenza reale", "Tempo di inferenza (s)", "RUN_BACKTEST",
    "N_BACKTEST_ORIGINS", "SHRINKAGE_ENABLED", "q globale (mediana shrinkage)",
    "KPI Motul pesato", "SKU con risultato di backtest", "SKU esclusi dal backtest",
    "SKU con accuratezza nulla", "SKU senza forecast", "BUSINESS_ADJUSTMENT_FACTOR",
    "ROUNDING_MODE",
]


def _run_info_table(base, new):
    if not base.run_info and not new.run_info:
        return ["(foglio \"Run info\" assente in entrambe le run)"]
    rows = []
    for field in RUN_INFO_FIELDS:
        left = base.run_info.get(field, "")
        right = new.run_info.get(field, "")
        if left == "" and right == "":
            continue
        rows.append([field, left, right])
    return _top_table(rows, ["Campo", base.label, new.label])


def build_report(base, new, gates, mode):
    lines = [
        "# Confronto fra due run del forecast",
        "",
        f"- Modo: **{mode}**",
        f"- Baseline: `{base.label}`",
        f"- Nuovo:    `{new.label}`",
        f"- SKU nella tabella finale: {len(base.data)} -> {len(new.data)}",
        f"- Colonne forecast: {len(base.forecast_cols)} -> {len(new.forecast_cols)}",
        "",
        "## Gate",
        "",
    ]
    lines.extend(_top_table(
        [[g.name, g.title, g.verdict] for g in gates],
        ["Gate", "Descrizione", "Esito"],
    ))
    lines.append("")
    for gate in gates:
        lines.append(f"### {gate.name} — {gate.title} ({gate.verdict})")
        lines.extend(f"- {d}" for d in gate.details)
        lines.append("")

    lines.append("## Diagnostica")
    lines.append("")
    lines.extend(diagnostics(base, new))
    lines.append("")

    for title, table in deviation_tables(base, new):
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(table)
        lines.append("")

    lines.append("## Run info")
    lines.append("")
    lines.extend(_run_info_table(base, new))
    lines.append("")

    failed = [g.name for g in gates if g.applicable and not g.passed]
    if failed:
        lines.append(f"**Gate non superati: {', '.join(failed)}.** "
                     f"G1 e G2 falliti sono un bug del refactor e non hanno "
                     f"uscite negoziabili; G3, G4 o G5 falliti aprono l'indagine "
                     f"di § 9.3, la cui decisione e' dell'utente.")
    else:
        lines.append("**Tutti i gate applicabili sono soddisfatti.**")

    return "\n".join(lines)


def run_gates(base, new, mode, thresholds):
    if mode == "refactor":
        return [gate_g1(base, new)]
    gates = [gate_g2(base, new),
             gate_g3(base, new, thresholds["g3"]),
             gate_g4(base, new, thresholds["g4"])]
    if mode == "batch":
        gates.append(gate_g5(base, new))
    return gates


def compare(baseline, new, mode="batch", thresholds=None):
    thresholds = thresholds or dict(DEFAULT_THRESHOLDS[mode])
    gates = run_gates(baseline, new, mode, thresholds)
    report = build_report(baseline, new, gates, mode)
    ok = all(g.passed for g in gates if g.applicable)
    return ok, report, gates


# ======================================================================
# CLI
# ======================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Confronta due run del forecast e calcola i gate G1-G5 (§ 9.2).",
    )
    parser.add_argument("--baseline", required=True, help="Excel della run di riferimento")
    parser.add_argument("--new", required=True, help="Excel della run da valutare")
    parser.add_argument("--baseline-backtest", help="CSV di backtest della baseline")
    parser.add_argument("--new-backtest", help="CSV di backtest della run nuova")
    parser.add_argument("--baseline-errors", help="CSV degli SKU falliti (baseline)")
    parser.add_argument("--new-errors", help="CSV degli SKU falliti (run nuova)")
    parser.add_argument("--mode", choices=sorted(DEFAULT_THRESHOLDS),
                        default="batch",
                        help="refactor = T4.1 (G1); batch = T4.2 (G2-G5); "
                             "nobacktest = T4b (G2-G4, soglie strette)")
    parser.add_argument("--id-col", default="SKU")
    parser.add_argument("--g3-threshold", type=float,
                        help="sovrascrive la soglia di G3 (frazione, es. 0.005)")
    parser.add_argument("--g4-threshold", type=float,
                        help="sovrascrive la soglia di G4 (frazione, es. 0.01)")
    parser.add_argument("--report", help="scrive il report anche su questo file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    baseline = load_run(os.path.basename(args.baseline), args.baseline,
                        args.baseline_backtest, args.baseline_errors, args.id_col)
    new = load_run(os.path.basename(args.new), args.new,
                   args.new_backtest, args.new_errors, args.id_col)

    thresholds = dict(DEFAULT_THRESHOLDS[args.mode])
    if args.g3_threshold is not None:
        thresholds["g3"] = args.g3_threshold
    if args.g4_threshold is not None:
        thresholds["g4"] = args.g4_threshold

    ok, report, _gates = compare(baseline, new, args.mode, thresholds)

    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
        print(f"\nReport scritto in {args.report}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

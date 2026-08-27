"""
Test dell'utility di confronto `tests/tools/compare_forecast_outputs.py`.

Non e' fra i file elencati dalla Fase 4 del piano, ma i gate G1-G5 sono lo
strumento con cui si decide se il refactor e' accettabile: un'utility che li
calcola in silenzio nel modo sbagliato renderebbe verde un collaudo che non lo e'.
Qui si verifica che ogni gate scatti quando deve e solo quando deve, su
artefatti costruiti apposta.
"""

import os

import pandas as pd
import pytest

from forecast_lib.export import save_excel
from tests.tools.compare_forecast_outputs import (
    _fmt_cell,
    _frames_equal,
    compare,
    load_run,
    main,
)


FORECAST_COLS = ["f2026_01", "f2026_02"]


def write_run(tmp_path, name, rows, run_info=None, backtest=None, errors=None):
    """Scrive gli artefatti di una run (Excel + CSV di audit) e restituisce i path."""
    df = (pd.DataFrame.from_dict(rows, orient="index")
          .reset_index().rename(columns={"index": "SKU"}))
    xlsx = tmp_path / f"{name}.xlsx"
    save_excel(df, str(xlsx), run_info=run_info)

    paths = {"xlsx_path": str(xlsx), "backtest_csv": None, "errors_csv": None}
    if backtest is not None:
        bt = tmp_path / f"{name} backtest.csv"
        pd.DataFrame(backtest).to_csv(bt, index=False)
        paths["backtest_csv"] = str(bt)
    if errors is not None:
        err = tmp_path / f"{name} errori.csv"
        pd.DataFrame({"SKU": list(errors),
                      "Errore": ["forecast non disponibile"] * len(errors)}
                     ).to_csv(err, index=False)
        paths["errors_csv"] = str(err)
    return paths


def base_rows(volumes=None, abc=None):
    volumes = volumes or {"A": 1000.0, "B": 500.0, "C": 100.0}
    abc = abc or {"A": "A", "B": "B", "C": "C"}
    return {
        sku: {
            "Description": f"desc {sku}",
            "ABC": abc[sku],
            "XYZ": "X",
            "SafetyStock": 10.0,
            "LT": 30,
            "f2026_01": volume / 2,
            "f2026_02": volume / 2,
        }
        for sku, volume in volumes.items()
    }


def base_backtest(quantiles=None, accuracy=0.7):
    quantiles = quantiles or {"A": 0.50, "B": 0.55, "C": 0.60}
    return {
        "SKU": list(quantiles),
        "BestQuantile": list(quantiles.values()),
        "BestQuantileRaw": list(quantiles.values()),
        "BestAccuracy": [accuracy] * len(quantiles),
        "BestAccuracyRaw": [accuracy] * len(quantiles),
        "TotalWeight": [1000.0, 500.0, 100.0][: len(quantiles)],
        "q_global": [0.57] * len(quantiles),
    }


def make_pair(tmp_path, new_rows=None, new_backtest=None, base_run_kwargs=None,
              new_run_kwargs=None):
    base_kwargs = {"rows": base_rows(), "backtest": base_backtest(),
                   "errors": ["Z"]}
    base_kwargs.update(base_run_kwargs or {})
    new_kwargs = {"rows": new_rows if new_rows is not None else base_rows(),
                  "backtest": new_backtest if new_backtest is not None
                  else base_backtest(),
                  "errors": ["Z"]}
    new_kwargs.update(new_run_kwargs or {})

    base_paths = write_run(tmp_path, "run_base", **base_kwargs)
    new_paths = write_run(tmp_path, "run_nuovo", **new_kwargs)
    return (load_run("base", **base_paths), load_run("nuovo", **new_paths))


def gates_by_name(gates):
    return {g.name: g for g in gates}


# ----------------------------------------------------------------------
# G1 — identita' del refactor
# ----------------------------------------------------------------------

def test_g1_passes_on_identical_runs(tmp_path):
    base, new = make_pair(tmp_path)
    ok, report, gates = compare(base, new, mode="refactor")
    assert ok is True
    assert gates_by_name(gates)["G1"].passed is True
    assert "bit-identico" in report


def test_g1_fails_on_a_single_changed_cell(tmp_path):
    rows = base_rows()
    rows["B"]["f2026_02"] = rows["B"]["f2026_02"] + 0.001
    base, new = make_pair(tmp_path, new_rows=rows)

    ok, report, gates = compare(base, new, mode="refactor")
    assert ok is False
    assert gates_by_name(gates)["G1"].passed is False
    assert "B / f2026_02" in report


def test_g1_is_nan_aware(tmp_path):
    """`build_final_table` fa merge how='left': gli SKU senza forecast hanno NaN.
    Due NaN devono contare come uguali, non come differenti."""
    rows = base_rows()
    rows["C"]["f2026_01"] = float("nan")
    rows["C"]["f2026_02"] = float("nan")
    base, new = make_pair(tmp_path, new_rows=rows,
                          base_run_kwargs={"rows": rows})

    ok, _report, gates = compare(base, new, mode="refactor")
    assert ok is True
    assert gates_by_name(gates)["G1"].passed is True


def test_g1_fails_when_a_nan_becomes_a_number(tmp_path):
    base_data = base_rows()
    base_data["C"]["f2026_01"] = float("nan")
    new_data = base_rows()
    base, new = make_pair(tmp_path, new_rows=new_data,
                          base_run_kwargs={"rows": base_data})

    ok, _report, gates = compare(base, new, mode="refactor")
    assert ok is False


def test_g1_tolerates_different_backtest_column_sets(tmp_path):
    """Il CSV della run A non ha BestQuantileRaw/BestAccuracyRaw/q_global:
    arrivano con la Fase 1.5. Si confronta l'intersezione."""
    old_style = {
        "SKU": ["A", "B", "C"],
        "BestQuantile": [0.50, 0.55, 0.60],
        "BestAccuracy": [0.7, 0.7, 0.7],
        "TotalWeight": [1000.0, 500.0, 100.0],
    }
    base_paths = write_run(tmp_path, "run_a", rows=base_rows(), backtest=old_style)
    new_paths = write_run(tmp_path, "run_b", rows=base_rows(), backtest=base_backtest())
    base, new = load_run("run A", **base_paths), load_run("run B", **new_paths)

    ok, report, _gates = compare(base, new, mode="refactor")
    assert ok is True
    assert "ignorate" in report
    assert "BestQuantileRaw" in report


def test_g1_fails_when_the_backtest_csv_differs(tmp_path):
    base, new = make_pair(
        tmp_path, new_backtest=base_backtest({"A": 0.50, "B": 0.65, "C": 0.60}))
    ok, report, _gates = compare(base, new, mode="refactor")
    assert ok is False
    assert "DIFFERENZE nel CSV di backtest" in report


# ----------------------------------------------------------------------
# G2 — identita' strutturale
# ----------------------------------------------------------------------

def test_g2_passes_when_structure_is_identical(tmp_path):
    base, new = make_pair(tmp_path)
    _ok, _report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G2"].passed is True


def test_g2_fails_when_abc_changes(tmp_path):
    rows = base_rows(abc={"A": "A", "B": "C", "C": "C"})
    base, new = make_pair(tmp_path, new_rows=rows)
    _ok, report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G2"].passed is False
    assert "ABC: DIVERSA" in report


def test_g2_fails_when_an_sku_disappears_from_the_backtest(tmp_path):
    """Uno SKU che sparisce dal backtest ricade su q = 0.5 nel Modulo H:
    fino a 5x di scostamento. Non e' cosmetico."""
    reduced = base_backtest({"A": 0.50, "B": 0.55})
    base, new = make_pair(tmp_path, new_backtest=reduced)
    _ok, report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G2"].passed is False
    assert "SKU con risultato di backtest DIVERSI" in report


def test_g2_fails_when_the_failed_sku_set_changes(tmp_path):
    base, new = make_pair(tmp_path, new_run_kwargs={"errors": ["Z", "Y"]})
    _ok, report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G2"].passed is False
    assert "SKU senza forecast DIVERSI" in report


def test_g2_reports_when_the_audit_csvs_are_missing(tmp_path):
    base_paths = write_run(tmp_path, "a", rows=base_rows())
    new_paths = write_run(tmp_path, "b", rows=base_rows())
    base, new = load_run("a", **base_paths), load_run("b", **new_paths)

    ok, report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G2"].passed is True
    assert "non forniti" in report
    # senza BestQuantile, G5 non e' valutabile e non deve far fallire il confronto
    assert gates_by_name(gates)["G5"].applicable is False
    assert ok is True


# ----------------------------------------------------------------------
# G3 / G4 — impatto aggregato
# ----------------------------------------------------------------------

def test_g3_passes_within_the_threshold(tmp_path):
    rows = base_rows({"A": 1004.0, "B": 500.0, "C": 100.0})   # +0.25% sul totale
    base, new = make_pair(tmp_path, new_rows=rows)
    _ok, _report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G3"].passed is True


def test_g3_fails_beyond_the_threshold(tmp_path):
    rows = base_rows({"A": 1100.0, "B": 500.0, "C": 100.0})   # +6.25%
    base, new = make_pair(tmp_path, new_rows=rows)
    _ok, report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G3"].passed is False
    assert "+6.2500%" in report


def test_g4_catches_a_compensating_redistribution(tmp_path):
    """G3 ha segno e si compensa: una redistribuzione ampia fra SKU passerebbe
    indenne. E' esattamente il caso per cui esiste G4."""
    rows = base_rows({"A": 1050.0, "B": 450.0, "C": 100.0})   # totale invariato
    base, new = make_pair(tmp_path, new_rows=rows)
    _ok, _report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G3"].passed is True
    assert gates_by_name(gates)["G4"].passed is False


def test_nobacktest_mode_uses_tighter_thresholds(tmp_path):
    """T4b: senza backtest non ci sono flip di q, restano solo gli
    arrotondamenti. Uno scostamento che in T4.2 passa, qui deve fallire."""
    rows = base_rows({"A": 1004.0, "B": 500.0, "C": 100.0})   # +0.25%
    base, new = make_pair(tmp_path, new_rows=rows)

    _ok, _report, batch_gates = compare(base, new, mode="batch")
    assert gates_by_name(batch_gates)["G3"].passed is True

    ok, _report, gates = compare(base, new, mode="nobacktest")
    assert gates_by_name(gates)["G3"].passed is False
    assert ok is False
    assert "G5" not in gates_by_name(gates)     # non applicabile senza backtest


# ----------------------------------------------------------------------
# G5 — spiegabilita'
# ----------------------------------------------------------------------

def test_g5_fails_on_a_large_deviation_without_a_quantile_change(tmp_path):
    rows = base_rows({"A": 1000.0, "B": 500.0, "C": 130.0})   # +30% su C
    base, new = make_pair(tmp_path, new_rows=rows)            # stessi q
    _ok, report, gates = compare(base, new, mode="batch")
    g5 = gates_by_name(gates)["G5"]
    assert g5.passed is False
    assert "C:" in report


def test_g5_passes_when_the_deviation_is_explained_by_a_quantile_flip(tmp_path):
    rows = base_rows({"A": 1000.0, "B": 500.0, "C": 130.0})
    flipped = base_backtest({"A": 0.50, "B": 0.55, "C": 0.10})
    base, new = make_pair(tmp_path, new_rows=rows, new_backtest=flipped)
    _ok, _report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G5"].passed is True


def test_g5_ignores_small_deviations(tmp_path):
    rows = base_rows({"A": 1000.0, "B": 500.0, "C": 104.0})   # +4%, sotto il 5%
    base, new = make_pair(tmp_path, new_rows=rows)
    _ok, _report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G5"].passed is True


# ----------------------------------------------------------------------
# Diagnostica e Run info
# ----------------------------------------------------------------------

def test_run_info_sheet_is_excluded_from_the_comparison_but_reported(tmp_path):
    """Il foglio contiene data/ora e non esiste nella run A: confrontarlo
    farebbe fallire G1 su ogni run. I suoi campi servono pero' al report."""
    base_paths = write_run(tmp_path, "a", rows=base_rows(), backtest=base_backtest(),
                           run_info={"Data e ora del run": "2026-08-27 10:00:00",
                                     "Device": "cuda (RTX 2070 SUPER)",
                                     "INFERENCE_BATCH_SIZE": 1})
    new_paths = write_run(tmp_path, "b", rows=base_rows(), backtest=base_backtest(),
                          run_info={"Data e ora del run": "2026-08-27 11:00:00",
                                    "Device": "cuda (RTX 2070 SUPER)",
                                    "INFERENCE_BATCH_SIZE": 32})
    base, new = load_run("run A", **base_paths), load_run("run B", **new_paths)

    ok, report, _gates = compare(base, new, mode="refactor")
    assert ok is True                                   # l'orario diverso non conta
    assert "INFERENCE_BATCH_SIZE" in report
    assert "RTX 2070 SUPER" in report


def test_diagnostics_report_quantile_changes_and_kpi(tmp_path):
    flipped = base_backtest({"A": 0.50, "B": 0.55, "C": 0.10})
    base, new = make_pair(tmp_path, new_backtest=flipped)
    _ok, report, _gates = compare(base, new, mode="batch")

    assert "SKU con `BestQuantile` diverso: **1** su 3" in report
    assert "KPI Motul pesato" in report
    assert "q_global" in report
    assert "Top-20 per scostamento relativo" in report
    assert "classe A o B" in report


def test_diagnostics_count_zero_accuracy_skus(tmp_path):
    zero_acc = base_backtest()
    zero_acc["BestAccuracyRaw"] = [0.0, 0.0, 0.7]
    base, new = make_pair(tmp_path, new_backtest=zero_acc)
    _ok, report, _gates = compare(base, new, mode="batch")
    assert "`BestAccuracyRaw == 0`" in report


def test_report_lists_class_ab_skus_over_four_percent(tmp_path):
    rows = base_rows({"A": 1000.0, "B": 530.0, "C": 100.0})   # +6% su B (classe B)
    flipped = base_backtest({"A": 0.50, "B": 0.60, "C": 0.60})
    base, new = make_pair(tmp_path, new_rows=rows, new_backtest=flipped)
    _ok, report, _gates = compare(base, new, mode="batch")

    section = report.split("classe A o B")[1]
    assert "| B |" in section


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_main_returns_zero_when_every_gate_passes(tmp_path, capsys):
    base_paths = write_run(tmp_path, "a", rows=base_rows(), backtest=base_backtest())
    new_paths = write_run(tmp_path, "b", rows=base_rows(), backtest=base_backtest())
    report_path = tmp_path / "report.md"

    code = main([
        "--baseline", base_paths["xlsx_path"], "--new", new_paths["xlsx_path"],
        "--baseline-backtest", base_paths["backtest_csv"],
        "--new-backtest", new_paths["backtest_csv"],
        "--mode", "refactor", "--report", str(report_path),
    ])

    assert code == 0
    assert "G1" in capsys.readouterr().out
    assert report_path.read_text(encoding="utf-8").startswith("# Confronto")


def test_main_returns_one_when_a_gate_fails(tmp_path, capsys):
    rows = base_rows({"A": 1200.0, "B": 500.0, "C": 100.0})
    base_paths = write_run(tmp_path, "a", rows=base_rows(), backtest=base_backtest())
    new_paths = write_run(tmp_path, "b", rows=rows, backtest=base_backtest())

    code = main([
        "--baseline", base_paths["xlsx_path"], "--new", new_paths["xlsx_path"],
        "--baseline-backtest", base_paths["backtest_csv"],
        "--new-backtest", new_paths["backtest_csv"],
        "--mode", "batch",
    ])

    assert code == 1
    assert "Gate non superati" in capsys.readouterr().out


def test_threshold_overrides_are_honoured(tmp_path, capsys):
    rows = base_rows({"A": 1100.0, "B": 500.0, "C": 100.0})
    base_paths = write_run(tmp_path, "a", rows=base_rows(), backtest=base_backtest())
    new_paths = write_run(tmp_path, "b", rows=rows,
                          backtest=base_backtest({"A": 0.60, "B": 0.55, "C": 0.60}))

    code = main([
        "--baseline", base_paths["xlsx_path"], "--new", new_paths["xlsx_path"],
        "--baseline-backtest", base_paths["backtest_csv"],
        "--new-backtest", new_paths["backtest_csv"],
        "--mode", "batch", "--g3-threshold", "0.10", "--g4-threshold", "0.10",
    ])

    assert code == 0


def test_load_run_rejects_a_file_whose_first_sheet_is_run_info(tmp_path):
    path = tmp_path / "invertito.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Campo": ["x"], "Valore": ["y"]}).to_excel(
            writer, sheet_name="Run info", index=False)
        pd.DataFrame({"SKU": ["A"], "f2026_01": [1.0]}).to_excel(
            writer, sheet_name="Sheet1", index=False)

    with pytest.raises(SystemExit, match="ordine dei fogli"):
        load_run("invertito", str(path))


# ----------------------------------------------------------------------
# Robustezza del confronto cella per cella
# ----------------------------------------------------------------------

def test_frames_equal_ignores_a_dtype_only_difference():
    """`DataFrame.equals` confronta anche i dtype: la stessa colonna letta una
    volta come int64 e una come float64 — capita a ogni giro di Excel, basta un
    NaN in un'altra riga — risulterebbe diversa pur essendo identica."""
    a = pd.DataFrame({"LT": [30, 60]}, index=["A", "B"], dtype="int64")
    b = pd.DataFrame({"LT": [30.0, 60.0]}, index=["A", "B"], dtype="float64")
    assert a.equals(b) is False
    assert _frames_equal(a, b) is True


def test_frames_equal_still_catches_the_last_bit():
    a = pd.DataFrame({"f2026_01": [1.0]}, index=["A"])
    b = pd.DataFrame({"f2026_01": [1.0 + 2 ** -50]}, index=["A"])
    assert _frames_equal(a, b) is False


def test_frames_equal_treats_two_nan_as_equal():
    a = pd.DataFrame({"f2026_01": [float("nan"), 1.0]}, index=["A", "B"])
    b = pd.DataFrame({"f2026_01": [float("nan"), 1.0]}, index=["A", "B"])
    assert _frames_equal(a, b) is True


def test_differences_on_text_columns_are_listed(tmp_path):
    """ABC e XYZ sono stringhe: un confronto che le passa da `to_numeric` le
    riduce a NaN e dichiara 'diversa' senza saper dire dove."""
    rows = base_rows(abc={"A": "A", "B": "C", "C": "C"})
    base, new = make_pair(tmp_path, new_rows=rows)
    _ok, report, _gates = compare(base, new, mode="batch")
    assert "B / ABC: 'B' -> 'C'" in report


def test_numeric_cells_are_printed_as_numbers():
    assert _fmt_cell(pd.Series([180.0]).iloc[0]) == "180.0"
    assert _fmt_cell(pd.Series([3]).iloc[0]) == "3"
    assert _fmt_cell("C") == "'C'"


# ----------------------------------------------------------------------
# CSV di audit degradati
# ----------------------------------------------------------------------

def test_an_empty_errors_csv_is_not_a_failure(tmp_path):
    """Il caso normale e' proprio questo: quasi nessun run ha SKU falliti, e un
    file vuoto non deve far morire il confronto dopo tutto il lavoro."""
    paths = write_run(tmp_path, "a", rows=base_rows(), backtest=base_backtest())
    empty = tmp_path / "vuoto.csv"
    empty.write_text(os.linesep, encoding="utf-8")

    run = load_run("a", paths["xlsx_path"], paths["backtest_csv"], str(empty))
    assert run.error_skus == set()


def test_an_errors_csv_with_only_the_header_is_read_as_no_errors(tmp_path):
    paths = write_run(tmp_path, "a", rows=base_rows(), backtest=base_backtest())
    header_only = tmp_path / "solo_intestazione.csv"
    header_only.write_text("SKU,Errore" + os.linesep, encoding="utf-8")

    run = load_run("a", paths["xlsx_path"], paths["backtest_csv"], str(header_only))
    assert run.error_skus == set()


def test_two_runs_without_failed_skus_pass_g2(tmp_path):
    base_paths = write_run(tmp_path, "a", rows=base_rows(),
                           backtest=base_backtest(), errors=[])
    new_paths = write_run(tmp_path, "b", rows=base_rows(),
                          backtest=base_backtest(), errors=[])
    base, new = load_run("a", **base_paths), load_run("b", **new_paths)

    _ok, report, gates = compare(base, new, mode="batch")
    assert gates_by_name(gates)["G2"].passed is True
    assert "SKU senza forecast: identici (0)" in report

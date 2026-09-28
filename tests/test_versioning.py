"""
T2 — Versioning, logica pura (veloce, tutto mockato).

Qui non gira nessun comando git vero: `_run_git` (o `subprocess.run`) e' sempre
sostituito. Serve a coprire il parsing, il confronto fra versioni, i messaggi e
le guardie degli orchestratori. La controparte su repo git reali e' in
`test_versioning_integration.py` (marcata `slow`): questi due file coprono cose
diverse e nessuno dei due sostituisce l'altro.
"""

import subprocess

import pytest

from forecast_lib import versioning
from forecast_lib.versioning import (
    _is_release_tag,
    _parse_ls_remote_tags,
    _parse_version,
    check_library_version,
    check_project_updates,
    check_timesfm_update,
    compare_versions,
    latest_lib_version,
    latest_timesfm_tag,
    local_repo_status,
    timesfm_tag,
)


class FakeCompleted:
    """Sostituto minimale di subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ----------------------------------------------------------------------
# timesfm_tag — unico punto in cui si aggiunge la 'v'
# ----------------------------------------------------------------------

def test_timesfm_tag_adds_prefix():
    assert timesfm_tag("2.0.2") == "v2.0.2"


def test_timesfm_tag_is_idempotent():
    # se la 'v' c'e' gia', non va raddoppiata
    assert timesfm_tag("v2.0.2") == "v2.0.2"


def test_timesfm_tag_strips_whitespace():
    assert timesfm_tag("  2.0.2  ") == "v2.0.2"


# ----------------------------------------------------------------------
# compare_versions
# ----------------------------------------------------------------------

def test_compare_versions_numeric_not_lexicographic():
    # il caso che un confronto fra stringhe sbaglierebbe
    assert compare_versions("2.0.10", "2.0.9") == 1
    assert compare_versions("2.0.9", "2.0.10") == -1


def test_compare_versions_ignores_v_prefix():
    assert compare_versions("v2.0.2", "2.0.2") == 0
    assert compare_versions("2.0.2", "v2.0.2") == 0


def test_compare_versions_minor_beats_patch():
    assert compare_versions("2.1.0", "2.0.2") == 1


def test_compare_versions_different_segment_counts():
    # i tag reali del repo TimesFM hanno lunghezze diverse (v1.2 accanto a v1.4.1)
    assert compare_versions("v1.2", "1.6.0") == -1
    assert compare_versions("1.5.1", "1.6.0") == -1
    assert compare_versions("v2", "2.0.0") == 0
    assert compare_versions("2.1", "2.0.9") == 1


def test_compare_versions_malformed_returns_none_without_raising():
    assert compare_versions("pippo", "2.0.2") is None
    assert compare_versions("2.0.2", "") is None
    assert compare_versions(None, "2.0.2") is None
    assert compare_versions("2.0.2", None) is None


def test_parse_version_stops_at_first_non_numeric_segment():
    # 'v2.0.2-rc1' -> il segmento finale non e' un intero puro ma inizia con una
    # cifra: viene letto come 2. 'v2.0.x' si ferma prima della x.
    assert _parse_version("v2.0.2-rc1") == [2, 0, 2]
    assert _parse_version("v2.0.x") == [2, 0]
    assert _parse_version("x.y.z") is None


# ----------------------------------------------------------------------
# Parsing di `git ls-remote --tags`
# ----------------------------------------------------------------------

def test_parse_ls_remote_tags_deduplicates_annotated_refs():
    # git emette due righe per ogni tag annotato: <ref> e <ref>^{}
    stdout = (
        "aaaaaaa\trefs/tags/v1.2.1\n"
        "bbbbbbb\trefs/tags/v1.2.1^{}\n"
        "ccccccc\trefs/tags/v2.0.2\n"
        "ddddddd\trefs/tags/v2.0.2^{}\n"
    )
    assert _parse_ls_remote_tags(stdout) == ["v1.2.1", "v2.0.2"]


def test_parse_ls_remote_tags_mixed_light_and_annotated():
    stdout = (
        "aaaaaaa\trefs/tags/v1.2\n"           # leggero
        "bbbbbbb\trefs/tags/v2.0.1\n"          # annotato
        "ccccccc\trefs/tags/v2.0.1^{}\n"
        "ddddddd\trefs/heads/master\n"         # non e' un tag: da ignorare
        "\n"                                   # riga vuota
        "malformata\n"                         # una sola colonna: da ignorare
    )
    assert _parse_ls_remote_tags(stdout) == ["v1.2", "v2.0.1"]


def test_parse_ls_remote_tags_empty_output():
    assert _parse_ls_remote_tags("") == []
    assert _parse_ls_remote_tags(None) == []


# ----------------------------------------------------------------------
# latest_timesfm_tag / latest_lib_version — casi degradati
# ----------------------------------------------------------------------

def test_latest_timesfm_tag_picks_highest_version(monkeypatch):
    stdout = (
        "a\trefs/tags/v1.2.1\n"
        "b\trefs/tags/v2.0.1\n"
        "c\trefs/tags/v2.0.2\n"
        "d\trefs/tags/v2.0.2^{}\n"
        "e\trefs/tags/v1.2.6\n"
    )
    monkeypatch.setattr(versioning, "_run_git",
                        lambda *a, **k: FakeCompleted(0, stdout))
    assert latest_timesfm_tag("https://example.invalid/repo.git") == "v2.0.2"


def test_latest_lib_version_strips_the_v(monkeypatch):
    monkeypatch.setattr(versioning, "_run_git",
                        lambda *a, **k: FakeCompleted(0, "a\trefs/tags/v1.6.0\n"))
    assert latest_lib_version("https://example.invalid/repo.git") == "1.6.0"


def test_latest_tag_ignores_non_version_tags(monkeypatch):
    stdout = (
        "a\trefs/tags/latest\n"
        "b\trefs/tags/release-candidate\n"
        "c\trefs/tags/v1.0.0\n"
    )
    monkeypatch.setattr(versioning, "_run_git",
                        lambda *a, **k: FakeCompleted(0, stdout))
    assert latest_timesfm_tag("https://example.invalid/repo.git") == "v1.0.0"


def test_release_tag_recognises_plain_versions():
    assert _is_release_tag("v2.0.2") is True
    assert _is_release_tag("2.0.2") is True
    assert _is_release_tag("v1.2") is True
    assert _is_release_tag("v3") is True


def test_release_tag_rejects_pre_releases_and_labels():
    assert _is_release_tag("v2.1.0rc1") is False
    assert _is_release_tag("2.0.2-rc1") is False
    assert _is_release_tag("v2.0.0-beta") is False
    assert _is_release_tag("latest") is False


def test_latest_tag_ignores_pre_release_tags(monkeypatch):
    """`_parse_version` leggerebbe 'v2.1.0rc1' come 2.1.0 e lo proporrebbe come
    aggiornamento: un release candidate non e' cio' che si consiglia a chi sta
    per rifare i forecast di produzione."""
    stdout = (
        "a\trefs/tags/v2.0.2\n"
        "b\trefs/tags/v2.1.0rc1\n"
        "c\trefs/tags/v2.2.0-beta\n"
    )
    monkeypatch.setattr(versioning, "_run_git",
                        lambda *a, **k: FakeCompleted(0, stdout))
    assert latest_timesfm_tag("https://example.invalid/repo.git") == "v2.0.2"


def test_latest_tag_on_timeout_returns_none(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(versioning, "_run_git", boom)
    assert latest_timesfm_tag("https://example.invalid/repo.git") is None
    assert latest_lib_version("https://example.invalid/repo.git") is None


def test_latest_tag_when_git_is_missing_returns_none(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git non installato")

    monkeypatch.setattr(versioning, "_run_git", boom)
    assert latest_timesfm_tag("https://example.invalid/repo.git") is None


def test_latest_tag_on_git_error_returns_none(monkeypatch):
    monkeypatch.setattr(versioning, "_run_git",
                        lambda *a, **k: FakeCompleted(128, "", "fatal: repository not found"))
    assert latest_timesfm_tag("https://example.invalid/repo.git") is None


def test_latest_tag_on_empty_output_returns_none(monkeypatch):
    monkeypatch.setattr(versioning, "_run_git", lambda *a, **k: FakeCompleted(0, ""))
    assert latest_timesfm_tag("https://example.invalid/repo.git") is None


def test_latest_tag_uses_ls_remote_with_terminal_prompt_disabled(monkeypatch):
    """`_run_git` deve comporre l'argv giusto e passare un ambiente EREDITATO
    con GIT_TERMINAL_PROMPT=0 (mai `env={}`: su Windows rompe il clone HTTPS)."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return FakeCompleted(0, "a\trefs/tags/v3.0.0\n")

    monkeypatch.setattr(versioning.subprocess, "run", fake_run)

    assert latest_timesfm_tag("https://example.invalid/repo.git") == "v3.0.0"
    assert seen["argv"] == ["git", "ls-remote", "--tags",
                            "https://example.invalid/repo.git"]
    env = seen["kwargs"]["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert len(env) > 1                       # ambiente ereditato, non vuoto
    assert seen["kwargs"]["timeout"] == versioning.CHECK_TIMEOUT


# ----------------------------------------------------------------------
# check_library_version — due messaggi distinti
# ----------------------------------------------------------------------

def test_check_library_version_equal_returns_none():
    assert check_library_version("1.6.0", "1.6.0") is None
    assert check_library_version("v1.6.0", "1.6.0") is None


def test_check_library_version_not_comparable_returns_none():
    assert check_library_version("pippo", "1.6.0") is None


def test_check_library_version_library_older_says_update_the_code():
    msg = check_library_version("1.5.1", "1.6.0")
    assert msg is not None
    assert "1.5.1" in msg and "1.6.0" in msg
    assert "git pull" in msg
    assert "notebook aggiornato" not in msg


def test_check_library_version_notebook_older_says_update_the_notebook():
    msg = check_library_version("1.6.0", "1.5.1")
    assert msg is not None
    assert "1.6.0" in msg and "1.5.1" in msg
    assert "riscarica il notebook" in msg
    assert "git pull" not in msg


def test_check_library_version_messages_differ_in_the_two_directions():
    older = check_library_version("1.5.1", "1.6.0")
    newer = check_library_version("1.6.0", "1.5.1")
    assert older != newer


# ----------------------------------------------------------------------
# local_repo_status — caso non-git (gli altri richiedono repo veri: T2b)
# ----------------------------------------------------------------------

def test_local_repo_status_on_non_git_directory_returns_none(tmp_path):
    assert local_repo_status(str(tmp_path), "https://example.invalid/repo.git") is None


def test_local_repo_status_on_missing_directory_returns_none(tmp_path):
    assert local_repo_status(str(tmp_path / "assente"), "https://x.invalid/r.git") is None


def test_local_repo_status_survives_a_fetch_timeout(tmp_path, monkeypatch):
    """Cinque secondi di budget per un fetch HTTPS si superano facilmente. Se il
    timeout diventasse `None`, il chiamante lo riporterebbe all'utente come
    "non e' un repository git" su un clone perfettamente valido."""
    (tmp_path / ".git").mkdir()
    remote = "https://example.invalid/repo.git"

    def fake_run_git(args, **kwargs):
        if args[0] == "fetch":
            raise subprocess.TimeoutExpired(cmd="git fetch", timeout=5)
        if args[:2] == ["remote", "get-url"]:
            return FakeCompleted(0, remote + "\n")
        if args[0] == "rev-parse":
            return FakeCompleted(0, "main\n")
        if args[0] == "status":
            return FakeCompleted(0, "")
        return FakeCompleted(0, "")

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)

    status = local_repo_status(str(tmp_path), remote)

    assert status is not None                 # NON e' "non e' un repository git"
    assert status["branch"] == "main"
    assert status["dirty"] is False
    assert status["remote_matches"] is True
    assert status["ahead"] is None            # semplicemente non calcolabili
    assert status["behind"] is None


def test_check_project_updates_does_not_claim_zip_install_on_a_fetch_timeout(
        tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    remote = "https://example.invalid/repo.git"

    def fake_run_git(args, **kwargs):
        if args[0] == "fetch":
            raise subprocess.TimeoutExpired(cmd="git fetch", timeout=5)
        if args[:2] == ["remote", "get-url"]:
            return FakeCompleted(0, remote + "\n")
        if args[0] == "rev-parse":
            return FakeCompleted(0, "main\n")
        return FakeCompleted(0, "")

    monkeypatch.setattr(versioning, "_run_git", fake_run_git)
    monkeypatch.setattr(versioning, "latest_lib_version", lambda *a, **k: None)

    msgs = check_project_updates(
        colab=False, enabled=True, repo_path=str(tmp_path), repo_url=remote,
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert not any("non e' un repository git" in m for m in msgs)


# ----------------------------------------------------------------------
# check_project_updates — guardie sui check di rete
# ----------------------------------------------------------------------

class NetworkSpy:
    """Registra le invocazioni dei check di rete e restituisce un valore fisso."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def spies(monkeypatch):
    latest = NetworkSpy(result=None)
    status = NetworkSpy(result=None)
    monkeypatch.setattr(versioning, "latest_lib_version", latest)
    monkeypatch.setattr(versioning, "local_repo_status", status)
    return {"latest_lib_version": latest, "local_repo_status": status}


def test_check_project_updates_disabled_does_no_network_at_all(spies):
    msgs = check_project_updates(
        colab=False, enabled=False,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert spies["latest_lib_version"].calls == []
    assert spies["local_repo_status"].calls == []
    assert msgs == []


def test_check_project_updates_disabled_still_compares_notebook_and_library(spies):
    """Il confronto notebook/libreria non fa I/O: gira anche con enabled=False."""
    msgs = check_project_updates(
        colab=False, enabled=False,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.5.1", expected_lib_version="1.6.0",
    )
    assert len(msgs) == 1
    assert "git pull" in msgs[0]
    assert spies["latest_lib_version"].calls == []


def test_check_project_updates_in_colab_skips_local_repo_status(spies):
    check_project_updates(
        colab=True, enabled=True,
        repo_path="/content/Forecast-TimesFM-and-SS",
        repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    # in Colab il clone e' fresco per costruzione: nessuno stato locale da leggere
    assert spies["local_repo_status"].calls == []
    # ma il confronto col remoto serve proprio li': il pin invecchia inosservato
    assert len(spies["latest_lib_version"].calls) == 1


def test_check_project_updates_local_runs_both_checks(spies):
    check_project_updates(
        colab=False, enabled=True,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert len(spies["latest_lib_version"].calls) == 1
    assert len(spies["local_repo_status"].calls) == 1


def test_check_project_updates_warns_when_remote_is_newer(monkeypatch, spies):
    spies["latest_lib_version"].result = "1.7.0"
    msgs = check_project_updates(
        colab=True, enabled=True,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert any("1.7.0" in m for m in msgs)


def test_check_project_updates_silent_when_remote_is_not_newer(spies):
    spies["latest_lib_version"].result = "1.6.0"
    msgs = check_project_updates(
        colab=True, enabled=True,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert msgs == []


def test_check_project_updates_reports_non_git_local_repo(spies):
    spies["local_repo_status"].result = None
    msgs = check_project_updates(
        colab=False, enabled=True,
        repo_path="/percorso/finto", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert any("non e' un repository git" in m for m in msgs)


def test_check_project_updates_reports_behind_dirty_and_wrong_remote(spies):
    spies["local_repo_status"].result = {
        "branch": "main", "dirty": True, "ahead": 0, "behind": 3,
        "remote": "https://example.invalid/fork.git", "remote_matches": False,
    }
    msgs = check_project_updates(
        colab=False, enabled=True,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    joined = "\n".join(msgs)
    assert "fork.git" in joined
    assert "indietro di 3 commit" in joined
    assert "modifiche non committate" in joined


def test_check_project_updates_never_raises(monkeypatch, spies):
    def boom(*a, **k):
        raise RuntimeError("rete esplosa")

    monkeypatch.setattr(versioning, "latest_lib_version", boom)
    msgs = check_project_updates(
        colab=True, enabled=True,
        repo_path=".", repo_url="https://example.invalid/repo.git",
        lib_version="1.6.0", expected_lib_version="1.6.0",
    )
    assert any("Controllo aggiornamenti non riuscito" in m for m in msgs)


# ----------------------------------------------------------------------
# check_timesfm_update
# ----------------------------------------------------------------------

def test_check_timesfm_update_disabled_does_no_network(monkeypatch):
    spy = NetworkSpy(result="v9.9.9")
    monkeypatch.setattr(versioning, "latest_timesfm_tag", spy)
    assert check_timesfm_update(
        enabled=False, repo_url="https://example.invalid/timesfm.git",
        current_version="2.0.2",
    ) is None
    assert spy.calls == []


def test_check_timesfm_update_warns_on_newer_tag(monkeypatch):
    monkeypatch.setattr(versioning, "latest_timesfm_tag", NetworkSpy(result="v2.1.0"))
    msg = check_timesfm_update(
        enabled=True, repo_url="https://example.invalid/timesfm.git",
        current_version="2.0.2",
    )
    assert msg is not None
    assert "v2.1.0" in msg and "v2.0.2" in msg
    # deve rimandare al runbook, non promettere un aggiornamento automatico
    assert "Aggiornare TimesFM" in msg
    assert "TIMESFM_MODEL_REVISION" in msg


def test_check_timesfm_update_silent_when_pin_is_current(monkeypatch):
    monkeypatch.setattr(versioning, "latest_timesfm_tag", NetworkSpy(result="v2.0.2"))
    assert check_timesfm_update(
        enabled=True, repo_url="https://example.invalid/timesfm.git",
        current_version="2.0.2",
    ) is None


def test_check_timesfm_update_silent_when_pin_is_newer_than_remote(monkeypatch):
    monkeypatch.setattr(versioning, "latest_timesfm_tag", NetworkSpy(result="v2.0.1"))
    assert check_timesfm_update(
        enabled=True, repo_url="https://example.invalid/timesfm.git",
        current_version="2.0.2",
    ) is None


def test_check_timesfm_update_silent_when_remote_unreachable(monkeypatch):
    monkeypatch.setattr(versioning, "latest_timesfm_tag", NetworkSpy(result=None))
    assert check_timesfm_update(
        enabled=True, repo_url="https://example.invalid/timesfm.git",
        current_version="2.0.2",
    ) is None


def test_check_timesfm_update_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rete esplosa")

    monkeypatch.setattr(versioning, "latest_timesfm_tag", boom)
    assert check_timesfm_update(
        enabled=True, repo_url="https://example.invalid/timesfm.git",
        current_version="2.0.2",
    ) is None

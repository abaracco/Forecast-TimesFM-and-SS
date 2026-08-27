"""
T2b — Versioning su repository git REALI (marcato `slow`).

Serve perche' `test_versioning.py` mocka `subprocess`: non avrebbe mai potuto
scoprire il problema dei filtri LFS del repo TimesFM, che e' proprio quello che
ha determinato la forma del clone (`--filter=blob:none --sparse` +
`sparse-checkout set src`).

Due famiglie di test:

  - la stragrande maggioranza gira contro un **finto repo TimesFM locale**
    (`git init` in una tmp_path): offline, veloce, e permette di provocare a
    comando i casi degradati (tag inesistente, remote diverso, clone fallito);
  - **un** test clona davvero `google-research/timesfm` al tag pinnato: e' il
    canary dei filtri LFS, l'unica cosa che un finto remote non puo' riprodurre.

Esecuzione: `pytest -m slow`.
"""

import os
import shutil
import subprocess

import pytest

from forecast_lib import versioning
from forecast_lib.versioning import (
    _is_clean,
    _head_commit,
    _tag_commit,
    ensure_timesfm_checkout,
    latest_lib_version,
    local_repo_status,
)

pytestmark = pytest.mark.slow


TIMESFM_URL = "https://github.com/google-research/timesfm.git"
TIMESFM_TAG = "v2.0.2"

# Identita' fissa per i commit dei repo di prova: la configurazione globale
# della macchina non deve influire sull'esito.
GIT_IDENTITY = [
    "-c", "user.name=Test",
    "-c", "user.email=test@example.invalid",
    "-c", "commit.gpgsign=false",
]


def git(*args, cwd=None, check=True):
    res = subprocess.run(["git", *GIT_IDENTITY, *args], cwd=cwd,
                         capture_output=True, text=True)
    if check and res.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} fallito ({res.returncode}):\n"
            f"{res.stderr or res.stdout}"
        )
    return res


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ======================================================================
# Finto repo TimesFM locale
# ======================================================================

@pytest.fixture(scope="module")
def fake_remote(tmp_path_factory):
    """Repo locale con la struttura minima che il loader si aspetta (`src/`) e
    due tag: v2.0.1 e v2.0.2 (annotato, cosi' `refs/tags/<t>^{commit}` non e' un
    caso banale)."""
    root = tmp_path_factory.mktemp("fake_timesfm_remote") / "timesfm"
    root.mkdir()
    git("init", "-q", "-b", "main", cwd=root)

    write(root / "README.md", "finto TimesFM\n")
    write(root / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py",
          "class TimesFM_2p5_200M_torch:\n    pass\n")
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "v2.0.1", cwd=root)
    git("tag", "v2.0.1", cwd=root)

    write(root / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py",
          "class TimesFM_2p5_200M_torch:\n    VERSION = '2.0.2'\n")
    git("commit", "-q", "-am", "v2.0.2", cwd=root)
    git("tag", "-a", "v2.0.2", "-m", "release 2.0.2", cwd=root)

    return str(root)


@pytest.fixture
def checkout(tmp_path, fake_remote):
    """Cartella di destinazione gia' clonata e pinnata a v2.0.2."""
    dest = tmp_path / "timesfm"
    ensure_timesfm_checkout(str(dest), fake_remote, "v2.0.2", strict=True)
    return dest


# ----------------------------------------------------------------------
# ensure_timesfm_checkout — percorso felice
# ----------------------------------------------------------------------

def test_clone_from_scratch_is_pinned_and_clean(tmp_path, fake_remote):
    dest = tmp_path / "timesfm"
    info = ensure_timesfm_checkout(str(dest), fake_remote, "v2.0.2", strict=True)

    assert info["action"] == "clone"
    assert info["pin_verified"] is True
    assert info["message"] is None
    assert info["head"] == _tag_commit(str(dest), "v2.0.2")
    assert (dest / "src").is_dir()

    # Il working tree deve restare pulito su controlli RIPETUTI e dopo un touch:
    # la stat-cache di git rende un singolo `status` inaffidabile.
    assert _is_clean(str(dest))
    assert _is_clean(str(dest))
    target = dest / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py"
    os.utime(target, None)
    assert _is_clean(str(dest))


def test_second_call_on_a_correct_checkout_touches_no_network(checkout, fake_remote,
                                                              monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("nessun clone deve partire su un checkout gia' corretto")

    monkeypatch.setattr(versioning, "_clone_pinned", forbidden)
    monkeypatch.setattr(versioning, "_clone_and_swap", forbidden)

    info = ensure_timesfm_checkout(str(checkout), fake_remote, "v2.0.2", strict=True)

    assert info["action"] == "reused"
    assert info["pin_verified"] is True
    assert _is_clean(str(checkout))


def test_checkout_on_a_different_tag_is_realigned(tmp_path, fake_remote):
    dest = tmp_path / "timesfm"
    ensure_timesfm_checkout(str(dest), fake_remote, "v2.0.1", strict=True)
    assert _head_commit(str(dest)) == _tag_commit(str(dest), "v2.0.1")

    info = ensure_timesfm_checkout(str(dest), fake_remote, "v2.0.2", strict=True)

    assert info["action"] == "reclone"
    assert info["pin_verified"] is True
    assert info["head"] == _tag_commit(str(dest), "v2.0.2")
    assert _is_clean(str(dest))
    assert "VERSION = '2.0.2'" in (
        dest / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py"
    ).read_text(encoding="utf-8")


def test_dirty_checkout_is_recloned(checkout, fake_remote):
    """Esercita anche il `rmtree` con handler read-only: su Windows i .pack e
    .idx di .git non sono cancellabili senza chmod."""
    target = checkout / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py"
    target.write_text("# modifica non committata\n", encoding="utf-8")
    assert not _is_clean(str(checkout))

    info = ensure_timesfm_checkout(str(checkout), fake_remote, "v2.0.2", strict=True)

    assert info["action"] == "reclone"
    assert info["pin_verified"] is True
    assert _is_clean(str(checkout))
    assert "TimesFM_2p5_200M_torch" in target.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# ensure_timesfm_checkout — casi degradati
# ----------------------------------------------------------------------

def test_non_git_directory_raises_and_is_left_alone(tmp_path, fake_remote):
    dest = tmp_path / "timesfm"
    dest.mkdir()
    (dest / "documento_dell_utente.txt").write_text("roba mia", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non e' un repository git"):
        ensure_timesfm_checkout(str(dest), fake_remote, "v2.0.2", strict=True)

    # mai cancellare la cartella di qualcun altro
    assert (dest / "documento_dell_utente.txt").exists()


def test_different_remote_raises_and_the_directory_survives(checkout, tmp_path):
    """Un fork E' un repo git: la guardia sul `.git` non basta, serve il
    controllo sul remote — e la cartella non va mai toccata."""
    other_url = "https://github.com/qualcun-altro/timesfm.git"

    with pytest.raises(RuntimeError, match="remote 'origin'"):
        ensure_timesfm_checkout(str(checkout), other_url, "v2.0.2", strict=True)

    assert checkout.is_dir()
    assert (checkout / "src").is_dir()
    assert _is_clean(str(checkout))


def test_failed_clone_leaves_the_existing_directory_in_place(checkout, tmp_path):
    """Verifica dello swap da temp: se il clone fallisce (rete giu' — lo scenario
    per cui esiste TIMESFM_PIN_STRICT = False) la cartella deve restare li'."""
    broken_url = str(tmp_path / "remote_inesistente.git")
    git("remote", "set-url", "origin", broken_url, cwd=checkout)
    # sporca il tree, cosi' il codice tenta la riclonazione
    (checkout / "src" / "timesfm" / "timesfm_2p5"
     / "timesfm_2p5_torch.py").write_text("# sporco\n", encoding="utf-8")

    info = ensure_timesfm_checkout(str(checkout), broken_url, "v2.0.2", strict=False)

    assert info["action"] == "kept"
    assert info["pin_verified"] is False
    assert info["message"] is not None
    assert checkout.is_dir()
    assert (checkout / "src").is_dir()
    # nessun residuo dello swap
    assert not any(p.name.startswith(".timesfm_tmp_")
                   for p in checkout.parent.iterdir())
    assert not any(".old_" in p.name for p in checkout.parent.iterdir())


def test_failed_clone_with_strict_raises_but_keeps_the_directory(checkout, tmp_path):
    broken_url = str(tmp_path / "remote_inesistente.git")
    git("remote", "set-url", "origin", broken_url, cwd=checkout)
    (checkout / "src" / "timesfm" / "timesfm_2p5"
     / "timesfm_2p5_torch.py").write_text("# sporco\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="TIMESFM_PIN_STRICT"):
        ensure_timesfm_checkout(str(checkout), broken_url, "v2.0.2", strict=True)

    assert checkout.is_dir()


def test_unknown_tag_raises_when_strict(checkout, fake_remote):
    with pytest.raises(RuntimeError, match="Versione TimesFM NON verificata"):
        ensure_timesfm_checkout(str(checkout), fake_remote, "v99.0.0", strict=True)


def test_unknown_tag_only_warns_when_not_strict(checkout, fake_remote, capsys):
    info = ensure_timesfm_checkout(str(checkout), fake_remote, "v99.0.0", strict=False)

    assert info["pin_verified"] is False
    assert "v99.0.0" in info["message"]
    assert "ATTENZIONE" in capsys.readouterr().out
    assert checkout.is_dir()


def test_deleted_src_is_restored_by_the_reclone(checkout, fake_remote):
    """Cancellare `src/` rende il tree sporco: il codice se ne accorge e riclona,
    invece di lasciare al loader un checkout inutilizzabile."""
    shutil.rmtree(checkout / "src")

    info = ensure_timesfm_checkout(str(checkout), fake_remote, "v2.0.2", strict=True)

    assert info["action"] == "reclone"
    assert (checkout / "src").is_dir()
    assert _is_clean(str(checkout))


def test_checkout_without_src_is_rejected(tmp_path):
    """Un checkout pinnato e pulito ma senza `src/` non e' utilizzabile dal
    loader: meglio un errore esplicito che un ImportError tre passi piu' avanti."""
    remote = tmp_path / "remote_senza_src"
    remote.mkdir()
    git("init", "-q", "-b", "main", cwd=remote)
    write(remote / "README.md", "niente src qui\n")
    git("add", "-A", cwd=remote)
    git("commit", "-q", "-m", "c1", cwd=remote)
    git("tag", "v2.0.2", cwd=remote)

    dest = tmp_path / "timesfm"
    with pytest.raises(RuntimeError, match="non contiene la cartella 'src'"):
        ensure_timesfm_checkout(str(dest), str(remote), "v2.0.2", strict=True)


# ----------------------------------------------------------------------
# Il clone vero: canary dei filtri LFS
# ----------------------------------------------------------------------

def test_real_timesfm_clone_is_pinned_and_the_tree_stays_clean(tmp_path):
    """L'unico test che tocca la rete davvero.

    E' l'unico modo di verificare che i filtri LFS del repo TimesFM non
    sporchino il working tree: su un finto remote quel problema non esiste, e
    con `subprocess` mockato non esisterebbe nemmeno il concetto.
    """
    dest = tmp_path / "timesfm"
    info = ensure_timesfm_checkout(str(dest), TIMESFM_URL, TIMESFM_TAG, strict=True)

    assert info["pin_verified"] is True
    assert info["head"] == _tag_commit(str(dest), TIMESFM_TAG)
    assert (dest / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py").is_file()

    assert _is_clean(str(dest))
    assert _is_clean(str(dest))
    os.utime(dest / "src" / "timesfm" / "timesfm_2p5" / "timesfm_2p5_torch.py", None)
    assert _is_clean(str(dest))


# ======================================================================
# local_repo_status su repo git veri (offline)
# ======================================================================

@pytest.fixture
def project_clone(tmp_path):
    """Finto 'origin' + clone locale, come il repo del progetto in locale."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    write(origin / "file.txt", "uno\n")
    git("add", "-A", cwd=origin)
    git("commit", "-q", "-m", "primo", cwd=origin)
    # un repo non-bare rifiuta il push sul branch corrente
    git("config", "receive.denyCurrentBranch", "ignore", cwd=origin)

    work = tmp_path / "work"
    git("clone", "-q", str(origin), str(work))
    remote = git("remote", "get-url", "origin", cwd=work).stdout.strip()
    return {"origin": origin, "work": work, "remote": remote}


def test_local_repo_status_on_a_fresh_clone(project_clone):
    status = local_repo_status(str(project_clone["work"]), project_clone["remote"])

    assert status["branch"] == "main"
    assert status["dirty"] is False
    assert status["ahead"] == 0
    assert status["behind"] == 0
    assert status["remote_matches"] is True


def test_local_repo_status_detects_behind(project_clone):
    write(project_clone["origin"] / "file.txt", "due\n")
    git("commit", "-q", "-am", "secondo", cwd=project_clone["origin"])

    status = local_repo_status(str(project_clone["work"]), project_clone["remote"])
    assert status["behind"] == 1
    assert status["ahead"] == 0


def test_local_repo_status_detects_ahead(project_clone):
    write(project_clone["work"] / "locale.txt", "mio\n")
    git("add", "-A", cwd=project_clone["work"])
    git("commit", "-q", "-m", "commit locale", cwd=project_clone["work"])

    status = local_repo_status(str(project_clone["work"]), project_clone["remote"])
    assert status["ahead"] == 1
    assert status["behind"] == 0


def test_local_repo_status_detects_dirty(project_clone):
    write(project_clone["work"] / "file.txt", "modificato\n")

    status = local_repo_status(str(project_clone["work"]), project_clone["remote"])
    assert status["dirty"] is True


def test_local_repo_status_on_detached_head(project_clone):
    write(project_clone["work"] / "secondo.txt", "x\n")
    git("add", "-A", cwd=project_clone["work"])
    git("commit", "-q", "-m", "secondo", cwd=project_clone["work"])
    git("checkout", "-q", "HEAD~1", cwd=project_clone["work"])

    status = local_repo_status(str(project_clone["work"]), project_clone["remote"])
    assert status["branch"] == "HEAD"
    assert status["ahead"] is None
    assert status["behind"] is None


def test_local_repo_status_on_a_branch_without_upstream(project_clone):
    git("checkout", "-q", "-b", "senza-upstream", cwd=project_clone["work"])

    status = local_repo_status(str(project_clone["work"]), project_clone["remote"])
    assert status["branch"] == "senza-upstream"
    assert status["ahead"] is None
    assert status["behind"] is None


def test_local_repo_status_does_not_fetch_when_the_remote_differs(project_clone,
                                                                  monkeypatch):
    """Il controllo sul remote precede QUALUNQUE contatto con la rete."""
    commands = []
    original = versioning._run_git

    def spy(args, **kwargs):
        commands.append(list(args))
        return original(args, **kwargs)

    monkeypatch.setattr(versioning, "_run_git", spy)

    status = local_repo_status(str(project_clone["work"]),
                               "https://example.invalid/altro.git")

    assert status["remote_matches"] is False
    assert status["ahead"] is None and status["behind"] is None
    assert not any(cmd[0] == "fetch" for cmd in commands)


def test_local_repo_status_on_a_non_git_directory(tmp_path):
    plain = tmp_path / "senza_git"
    plain.mkdir()
    assert local_repo_status(str(plain), "https://example.invalid/x.git") is None


# ======================================================================
# latest_lib_version contro un repo locale con tag misti
# ======================================================================

def test_latest_lib_version_against_a_real_repo_with_mixed_tags(tmp_path):
    repo = tmp_path / "tagged"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    write(repo / "f.txt", "x\n")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "c1", cwd=repo)

    git("tag", "v1.2", cwd=repo)                                  # leggero, 2 segmenti
    git("tag", "-a", "v1.5.1", "-m", "1.5.1", cwd=repo)           # annotato
    git("tag", "-a", "v1.6.0", "-m", "1.6.0", cwd=repo)           # annotato
    git("tag", "latest", cwd=repo)                                # non e' una versione

    assert latest_lib_version(str(repo)) == "1.6.0"


def test_latest_lib_version_on_a_repo_without_tags(tmp_path):
    repo = tmp_path / "senza_tag"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    write(repo / "f.txt", "x\n")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "c1", cwd=repo)

    assert latest_lib_version(str(repo)) is None


# ======================================================================
# Sequenza di allineamento del branch (cella 1 del notebook, § 2.3)
# ======================================================================

def align_to_branch(repo_path, branch):
    """Replica della sequenza della cella 1 del notebook.

    Va tenuta allineata a quella: e' l'unico punto in cui il notebook decide
    quale codice Colab eseguira'. Il clone e' `--depth 1` (quindi
    `--single-branch`), percio' ne' `git pull` ne' un fetch+checkout semplice
    bastano a passare a un altro branch.
    """
    cfg = git("config", "--get-all", "remote.origin.fetch",
              cwd=repo_path, check=False)
    if f"/{branch}:" not in cfg.stdout:
        git("remote", "set-branches", "--add", "origin", branch, cwd=repo_path)

    fetch_ok = git("fetch", "--depth", "1", "origin", branch,
                   cwd=repo_path, check=False).returncode == 0
    checkout_ok = git("checkout", "-B", branch, "FETCH_HEAD",
                      cwd=repo_path, check=False).returncode == 0
    git("branch", f"--set-upstream-to=origin/{branch}", branch,
        cwd=repo_path, check=False)
    return fetch_ok and checkout_ok


@pytest.fixture
def two_branch_clone(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    write(origin / "marker.txt", "main\n")
    git("add", "-A", cwd=origin)
    git("commit", "-q", "-m", "main", cwd=origin)

    git("checkout", "-q", "-b", "feat/timesfm", cwd=origin)
    write(origin / "marker.txt", "feat\n")
    git("commit", "-q", "-am", "feat", cwd=origin)
    git("checkout", "-q", "main", cwd=origin)
    git("config", "receive.denyCurrentBranch", "ignore", cwd=origin)

    work = tmp_path / "work"
    git("clone", "-q", "--depth", "1", "--branch", "main", str(origin), str(work))
    return {"origin": origin, "work": work}


def test_alignment_sequence_switches_branch_on_a_shallow_clone(two_branch_clone):
    work = two_branch_clone["work"]
    assert (work / "marker.txt").read_text(encoding="utf-8") == "main\n"

    assert align_to_branch(str(work), "feat/timesfm") is True

    assert git("rev-parse", "--abbrev-ref", "HEAD",
               cwd=work).stdout.strip() == "feat/timesfm"
    assert (work / "marker.txt").read_text(encoding="utf-8") == "feat\n"


def test_alignment_sequence_is_idempotent_and_does_not_duplicate_the_refspec(
        two_branch_clone):
    work = two_branch_clone["work"]
    align_to_branch(str(work), "feat/timesfm")

    refspecs = git("config", "--get-all", "remote.origin.fetch",
                   cwd=work).stdout.splitlines()
    assert sum(1 for r in refspecs if "/feat/timesfm:" in r) == 1

    assert align_to_branch(str(work), "feat/timesfm") is True

    refspecs_after = git("config", "--get-all", "remote.origin.fetch",
                         cwd=work).stdout.splitlines()
    assert refspecs_after == refspecs


def test_after_the_alignment_git_pull_works(two_branch_clone):
    """Il ramo alternativo 'se il branch e' gia' quello giusto basta git pull'
    e' proprio quello che falliva: senza upstream `git pull` muore con
    'refusing to merge unrelated histories'."""
    work = two_branch_clone["work"]
    align_to_branch(str(work), "feat/timesfm")

    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
                   cwd=work, check=False)
    assert upstream.returncode == 0
    assert upstream.stdout.strip() == "origin/feat/timesfm"

    pull = git("pull", "--depth", "1", cwd=work, check=False)
    assert pull.returncode == 0, pull.stderr


def test_alignment_sequence_reports_failure_on_a_missing_branch(two_branch_clone):
    """Il fetch non e' bloccante ma l'esito va verificato: se fallisce, Colab
    proseguirebbe in silenzio con il codice della sessione precedente."""
    work = two_branch_clone["work"]
    assert align_to_branch(str(work), "branch/inesistente") is False

"""
Versioning — pin del codice TimesFM e avvisi di aggiornamento.

Due famiglie di funzioni, con contratti opposti:

  1. VERIFICA DEL PIN (`ensure_timesfm_checkout`): porta ./timesfm al tag
     richiesto e verifica che ci sia davvero. Con `strict=True` (default)
     SOLLEVA se il pin non e' verificabile: e' l'unico punto del modulo che
     puo' interrompere l'esecuzione.

  2. CHECK INFORMATIVI (tutto il resto): dicono se esiste qualcosa di piu'
     recente, non aggiornano mai nulla e NON sollevano mai. Timeout corto,
     GIT_TERMINAL_PROMPT=0 per non restare appesi a un prompt di credenziali.

Note operative verificate sul campo (vedi PIANO_AGGIORNAMENTO_TIMESFM.md):
  - il clone deve essere `--filter=blob:none --sparse` + `sparse-checkout set src`:
    e' l'unica variante che lascia il working tree stabilmente pulito, perche' i
    file soggetti ai filtri LFS del repo TimesFM non esistono proprio;
  - i subprocess girano con l'ambiente EREDITATO (mai `env={}`: su Windows fa
    fallire il clone HTTPS con "Could not resolve host");
  - la directory temporanea per lo swap va creata accanto alla destinazione, non
    in %TEMP%: uno swap cross-volume fallisce;
  - su Windows i file .git/objects/pack/*.idx|.pack sono read-only e `rmtree`
    ha bisogno di un handler che faccia chmod.
"""

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile


# Timeout dei check informativi: devono essere impercettibili, non affidabili.
CHECK_TIMEOUT = 5
# I comandi che clonano hanno bisogno di tempo vero.
CLONE_TIMEOUT = 300
# Comandi git locali (rev-parse, status): nessuna rete di mezzo.
LOCAL_TIMEOUT = 30


# ----------------------------------------------------------------------
# Helper di basso livello
# ----------------------------------------------------------------------

def _git_env():
    """Ambiente ereditato + GIT_TERMINAL_PROMPT=0 (mai un env vuoto)."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(args, *, cwd=None, timeout=LOCAL_TIMEOUT):
    """Esegue `git *args` e restituisce il CompletedProcess. Non solleva mai
    per un exit code diverso da zero: la decisione spetta al chiamante."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_git_env(),
        timeout=timeout,
    )


def _rm_error_handler(func, path, _exc):
    """Handler per shutil.rmtree: su Windows i .pack/.idx di .git sono
    read-only e la cancellazione fallisce con PermissionError."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _rmtree(path):
    """rmtree tollerante ai file read-only (Windows)."""
    if not os.path.exists(path):
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_rm_error_handler)
    else:
        shutil.rmtree(path, onerror=_rm_error_handler)


def _normalize_url(url):
    """Normalizza un URL git per il confronto (suffisso .git, slash, case)."""
    if url is None:
        return ""
    s = str(url).strip().rstrip("/")
    if s.endswith(".git"):
        s = s[: -len(".git")]
    return s.lower()


# ----------------------------------------------------------------------
# Versioni e tag
# ----------------------------------------------------------------------

def timesfm_tag(version):
    """Tag git corrispondente a una versione TimesFM.

    UNICO punto in cui si aggiunge il prefisso 'v': `TIMESFM_VERSION` nel
    Modulo A e' sempre senza.
    """
    s = str(version).strip()
    return s if s.startswith("v") else f"v{s}"


def _parse_version(version):
    """'v2.0.10' -> [2, 0, 10]. None se non interpretabile."""
    if version is None:
        return None
    s = str(version).strip()
    if s[:1] in ("v", "V"):
        s = s[1:]
    parts = []
    for segment in s.split("."):
        m = re.match(r"^\s*(\d+)", segment)
        if not m:
            break
        parts.append(int(m.group(1)))
    return parts or None


def compare_versions(a, b):
    """Confronto numerico fra versioni: -1 se a < b, 0 se uguali, 1 se a > b.

    Tollera il prefisso 'v' e un numero di segmenti diverso ('v1.2' contro
    'v1.4.1': i tag reali del repo sono cosi'). Restituisce None se almeno una
    delle due non e' interpretabile — non solleva mai.
    """
    pa = _parse_version(a)
    pb = _parse_version(b)
    if pa is None or pb is None:
        return None
    n = max(len(pa), len(pb))
    pa = pa + [0] * (n - len(pa))
    pb = pb + [0] * (n - len(pb))
    return (pa > pb) - (pa < pb)


_RELEASE_TAG_RE = re.compile(r"^v?\d+(\.\d+)*$")


def _is_release_tag(tag):
    """True solo per i tag di rilascio veri: `v2.0.2`, `1.6`, `v3`.

    `_parse_version` e' volutamente tollerante (non deve mai sollevare su un tag
    strano), quindi da solo leggerebbe `v2.1.0rc1` come 2.1.0 e lo proporrebbe
    come aggiornamento disponibile. Un release candidate non e' una versione da
    consigliare a chi sta cercando la produzione.
    """
    return bool(_RELEASE_TAG_RE.match(str(tag).strip()))


def _parse_ls_remote_tags(stdout):
    """Estrae i nomi dei tag dall'output di `git ls-remote --tags`.

    Deduplica le righe 'refs/tags/<t>^{}' che git emette per i tag annotati:
    senza, un repo con tag misti restituirebbe ogni tag due volte.
    """
    tags = set()
    for line in (stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        ref = parts[-1].strip()
        if not ref.startswith("refs/tags/"):
            continue
        tag = ref[len("refs/tags/"):]
        if tag.endswith("^{}"):
            tag = tag[: -len("^{}")]
        if tag:
            tags.add(tag)
    return sorted(tags)


def _latest_tag(repo_url):
    """Tag di versione piu' alto su un repo remoto. None in ogni caso degradato
    (git assente, rete giu', timeout, nessun tag interpretabile)."""
    try:
        res = _run_git(["ls-remote", "--tags", repo_url], timeout=CHECK_TIMEOUT)
    except Exception:
        return None
    if res.returncode != 0:
        return None

    best = None
    for tag in _parse_ls_remote_tags(res.stdout):
        if not _is_release_tag(tag) or _parse_version(tag) is None:
            continue
        if best is None or compare_versions(tag, best) == 1:
            best = tag
    return best


def latest_timesfm_tag(repo_url):
    """Ultimo tag TimesFM disponibile (es. 'v2.0.2'), o None."""
    return _latest_tag(repo_url)


def latest_lib_version(repo_url):
    """Ultima versione di forecast_lib pubblicata come tag (es. '1.5.1'), o None.

    Serve a coprire il caso "installazione da ZIP", che
    EXPECTED_FORECAST_LIB_VERSION da solo non copre: uno ZIP e' coerente al
    proprio interno, quindi solo il confronto col remoto lo smaschera.
    """
    tag = _latest_tag(repo_url)
    if tag is None:
        return None
    return tag[1:] if tag[:1] in ("v", "V") else tag


def check_library_version(actual, expected):
    """Confronto puro fra la versione di forecast_lib e quella attesa dal
    notebook. None se coincidono (o se il confronto non e' possibile),
    altrimenti il messaggio con l'azione corretta — che e' diversa nei due
    versi: chi e' indietro non e' lo stesso componente."""
    cmp = compare_versions(actual, expected)
    if cmp is None or cmp == 0:
        return None
    if cmp < 0:
        return (
            f"forecast_lib e' piu' vecchia di quanto il notebook si aspetti "
            f"(installata {actual}, attesa {expected}): aggiorna il codice "
            f"(git pull, oppure riscarica lo ZIP del repository)."
        )
    return (
        f"Il notebook e' piu' vecchio del codice (forecast_lib {actual}, "
        f"il notebook si aspetta {expected}): riscarica il notebook aggiornato."
    )


# ----------------------------------------------------------------------
# Stato del repository locale
# ----------------------------------------------------------------------

def local_repo_status(repo_path, expected_remote):
    """Fotografia non invasiva di un checkout locale: mai pull, merge o checkout.

    Restituisce un dict con branch / dirty / ahead / behind / remote /
    remote_matches, oppure None se `repo_path` non e' un repo git.
    `ahead` e `behind` sono None quando non calcolabili (nessun upstream,
    HEAD staccata, fetch fallito, remote diverso da quello atteso).
    """
    try:
        if not os.path.isdir(os.path.join(repo_path, ".git")):
            return None

        res = _run_git(["remote", "get-url", "origin"], cwd=repo_path)
        remote = res.stdout.strip() if res.returncode == 0 else None

        res = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
        branch = res.stdout.strip() if res.returncode == 0 else None

        res = _run_git(["status", "--porcelain"], cwd=repo_path)
        dirty = bool(res.stdout.strip()) if res.returncode == 0 else None

        status = {
            "branch": branch,
            "dirty": dirty,
            "ahead": None,
            "behind": None,
            "remote": remote,
            "remote_matches": _normalize_url(remote) == _normalize_url(expected_remote),
        }

        # Nessun contatto con la rete se il remote non e' quello atteso.
        if not status["remote_matches"]:
            return status

        # Il fetch ha 5 secondi: su una connessione lenta li supera senza che
        # nulla sia rotto. Il timeout va assorbito QUI, non dall'except finale:
        # li' diventerebbe `None`, che il chiamante legge come "non e' un
        # repository git" e riporta all'utente come tale.
        try:
            res = _run_git(["fetch", "-q", "origin"], cwd=repo_path,
                           timeout=CHECK_TIMEOUT)
        except Exception:
            return status
        if res.returncode != 0:
            return status

        res = _run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_path,
        )
        if res.returncode != 0:
            return status  # HEAD staccata o branch senza upstream

        res = _run_git(["rev-list", "--left-right", "--count", "HEAD...@{u}"],
                       cwd=repo_path)
        if res.returncode == 0:
            parts = res.stdout.split()
            if len(parts) == 2:
                status["ahead"] = int(parts[0])
                status["behind"] = int(parts[1])
        return status
    except Exception:
        return None


# ----------------------------------------------------------------------
# Pin del checkout TimesFM
# ----------------------------------------------------------------------

def _tag_commit(repo_path, tag):
    """Commit a cui punta un tag nel repo locale, o None."""
    res = _run_git(["rev-parse", f"refs/tags/{tag}^{{commit}}"], cwd=repo_path)
    return res.stdout.strip() if res.returncode == 0 else None


def _head_commit(repo_path):
    res = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    return res.stdout.strip() if res.returncode == 0 else None


def _is_clean(repo_path):
    res = _run_git(["status", "--porcelain"], cwd=repo_path)
    return res.returncode == 0 and not res.stdout.strip()


def _verify_pin(repo_path, tag):
    """(pin_ok, head_commit, motivo_del_fallimento)."""
    head = _head_commit(repo_path)
    if head is None:
        return False, None, "HEAD non leggibile"
    tag_commit = _tag_commit(repo_path, tag)
    if tag_commit is None:
        return False, head, f"il tag {tag} non esiste nel checkout"
    if head != tag_commit:
        return False, head, f"HEAD ({head[:8]}) non coincide con {tag} ({tag_commit[:8]})"
    if not _is_clean(repo_path):
        return False, head, "il working tree contiene modifiche non committate"
    return True, head, None


def _clone_pinned(repo_url, tag, dest):
    """Clone sparse pinnato su un tag. Solleva se non riesce.

    Il fallback disattiva i filtri LFS: serve solo se il server non supporta
    il partial clone (`--filter=blob:none`). Il criterio di accettazione resta
    comunque la verifica del pin a valle.
    """
    args = [
        "clone", "-q", "--depth", "1", "--branch", tag,
        "--filter=blob:none", "--sparse", repo_url, dest,
    ]
    res = _run_git(args, timeout=CLONE_TIMEOUT)

    if res.returncode != 0:
        _rmtree(dest)  # un clone fallito puo' lasciare una directory parziale
        args = [
            "-c", "filter.lfs.smudge=",
            "-c", "filter.lfs.process=",
            "-c", "filter.lfs.required=false",
            "clone", "-q", "--depth", "1", "--branch", tag, repo_url, dest,
        ]
        res = _run_git(args, timeout=CLONE_TIMEOUT)
        if res.returncode != 0:
            raise RuntimeError(
                f"Clone di {repo_url} al tag {tag} fallito:\n"
                f"{(res.stderr or res.stdout or '').strip()}"
            )
        return "clone-lfs-disabled"

    # Solo src/ serve al loader (import relativi inclusi) e senza il resto il
    # working tree resta pulito.
    _run_git(["-C", dest, "sparse-checkout", "set", "src"], timeout=CLONE_TIMEOUT)
    return "clone-sparse"


def _clone_and_swap(repo_url, tag, path):
    """Riclona in una directory temporanea ACCANTO a `path` e scambia le due
    solo a clone riuscito: se la rete e' giu' la cartella esistente resta
    utilizzabile (e' lo scenario per cui esiste TIMESFM_PIN_STRICT = False)."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    tmp_root = tempfile.mkdtemp(dir=parent, prefix=".timesfm_tmp_")
    tmp_repo = os.path.join(tmp_root, "timesfm")
    backup = None
    try:
        _clone_pinned(repo_url, tag, tmp_repo)

        if os.path.exists(path):
            backup = f"{path}.old_{os.path.basename(tmp_root)}"
            os.rename(path, backup)
        try:
            os.replace(tmp_repo, path)
        except Exception:
            if backup is not None and not os.path.exists(path):
                os.rename(backup, path)  # ripristino: meglio il vecchio che niente
                backup = None
            raise
    finally:
        if backup is not None:
            _rmtree(backup)
        _rmtree(tmp_root)


def ensure_timesfm_checkout(timesfm_dir, repo_url, tag, strict=True):
    """Porta `timesfm_dir` al tag richiesto e verifica il pin.

    Restituisce un dict:
        {"path", "tag", "pin_verified", "head", "action", "message"}

    `action` vale "clone" (cartella assente), "reused" (gia' corretta),
    "reclone" (riclonata) o "kept" (riclonazione fallita, cartella preesistente
    lasciata al suo posto).

    Solleva:
      - se la cartella esiste ma non e' un repo git (va cancellata a mano:
        cancellarla noi sarebbe distruttivo su un percorso che, in
        JupyterLab/VS Code, dipende dalla CWD);
      - se il remote di origin non e' `repo_url` (un fork E' un repo git, quindi
        il controllo precedente non basta): mai cancellare la cartella di
        qualcun altro;
      - se al termine il pin non e' verificato e `strict=True`;
      - se non esiste alcun checkout utilizzabile.
    """
    path = os.path.abspath(timesfm_dir)
    action = None

    if not os.path.exists(path):
        print(f"Clone di TimesFM {tag} in {path} ...")
        _clone_pinned(repo_url, tag, path)
        action = "clone"
    elif not os.path.isdir(os.path.join(path, ".git")):
        raise RuntimeError(
            f"'{path}' esiste ma non e' un repository git. "
            f"Cancella la cartella e riesegui: verra' riclonata pinnata su {tag}."
        )
    else:
        res = _run_git(["remote", "get-url", "origin"], cwd=path)
        remote = res.stdout.strip() if res.returncode == 0 else None
        if _normalize_url(remote) != _normalize_url(repo_url):
            raise RuntimeError(
                f"'{path}' e' un repository git ma il suo remote 'origin' e' "
                f"{remote!r}, non {repo_url!r}. Non viene toccato: verifica di "
                f"essere nella directory giusta (in JupyterLab/VS Code la CWD "
                f"puo' non essere la root del progetto) e, se la cartella non "
                f"serve, spostala o cancellala a mano."
            )

        pin_ok, _head, reason = _verify_pin(path, tag)
        if pin_ok:
            action = "reused"
        else:
            print(f"Checkout TimesFM da riallineare a {tag} ({reason}). Riclono...")
            try:
                _clone_and_swap(repo_url, tag, path)
                action = "reclone"
            except Exception as exc:
                action = "kept"
                print(f"ATTENZIONE: riclonazione fallita ({exc}). "
                      f"La cartella esistente e' stata lasciata al suo posto.")

    pin_ok, head, reason = _verify_pin(path, tag)

    if not os.path.isdir(os.path.join(path, "src")):
        raise RuntimeError(
            f"Il checkout TimesFM in '{path}' non contiene la cartella 'src': "
            f"non e' utilizzabile. Cancella la cartella e riesegui."
        )

    message = None
    if not pin_ok:
        message = (
            f"Versione TimesFM NON verificata: {reason}. "
            f"I risultati di questo run non sono riproducibili."
        )
        if strict:
            raise RuntimeError(
                message
                + " (TIMESFM_PIN_STRICT = True). Cancella la cartella "
                  f"'{path}' e riesegui, oppure imposta TIMESFM_PIN_STRICT = False "
                  "per procedere consapevolmente senza pin verificato."
            )
        print("=" * 70)
        print(f"ATTENZIONE: {message}")
        print("=" * 70)
    else:
        print(f"TimesFM {tag} verificato ({head[:8]}), working tree pulito.")

    return {
        "path": path,
        "tag": tag,
        "pin_verified": pin_ok,
        "head": head,
        "action": action,
        "message": message,
    }


# ----------------------------------------------------------------------
# Orchestratori dei check informativi (mai bloccanti)
# ----------------------------------------------------------------------

def check_project_updates(*, colab, enabled, repo_path, repo_url,
                          lib_version, expected_lib_version):
    """Avvisi sullo stato di forecast_lib — cella 1. Non solleva mai.

    Il confronto notebook/libreria gira sempre (nessun I/O). I check di rete
    girano solo se `enabled`; `latest_lib_version` anche in Colab (e' li' che
    il codice rischia di invecchiare inosservato), `local_repo_status` no:
    in Colab il clone e' fresco per costruzione.

    Restituisce la lista dei messaggi emessi (vuota se va tutto bene).
    """
    messages = []
    try:
        msg = check_library_version(lib_version, expected_lib_version)
        if msg:
            messages.append(msg)

        if enabled:
            latest = latest_lib_version(repo_url)
            if latest is not None and compare_versions(latest, lib_version) == 1:
                messages.append(
                    f"Su GitHub esiste forecast_lib {latest} (in uso: {lib_version}). "
                    f"Aggiorna con `git pull` (o riscarica lo ZIP) e riavvia il kernel."
                )

        if enabled and not colab:
            status = local_repo_status(repo_path, repo_url)
            if status is None:
                messages.append(
                    f"'{repo_path}' non e' un repository git: non posso verificare "
                    f"se il codice e' aggiornato (installazione da ZIP?)."
                )
            else:
                if not status["remote_matches"]:
                    messages.append(
                        f"Il repository locale punta a {status['remote']!r}, non a "
                        f"{repo_url!r}: nessun controllo di allineamento eseguito."
                    )
                if status.get("behind"):
                    messages.append(
                        f"Il repository locale e' indietro di {status['behind']} commit "
                        f"su '{status['branch']}': valuta `git pull` e riavvia il kernel."
                    )
                if status.get("dirty"):
                    messages.append(
                        "Il repository locale ha modifiche non committate: "
                        "il codice in esecuzione non corrisponde ad alcuna versione pubblicata."
                    )
    except Exception as exc:  # un check informativo non ferma mai la pipeline
        messages.append(f"Controllo aggiornamenti non riuscito: {exc}")

    for msg in messages:
        print(f"AVVISO: {msg}")
    return messages


def check_timesfm_update(*, enabled, repo_url, current_version):
    """Avviso se esiste un tag TimesFM piu' recente di quello pinnato —
    cella 6, dopo `ensure_timesfm_checkout`. Non aggiorna nulla e non solleva.

    Attivo anche in Colab: essendo Colab la modalita' principale, e' li' che un
    pin invecchia senza che nessuno se ne accorga.

    Restituisce il messaggio emesso, oppure None.
    """
    if not enabled:
        return None
    try:
        latest = latest_timesfm_tag(repo_url)
        if latest is None:
            return None
        if compare_versions(latest, current_version) != 1:
            return None
        msg = (
            f"TimesFM {latest} e' disponibile (in uso: {timesfm_tag(current_version)}). "
            f"L'aggiornamento NON e' automatico: segui il runbook "
            f"\"Aggiornare TimesFM\" nel README (cambio di TIMESFM_VERSION, "
            f"rivalutazione di TIMESFM_MODEL_REVISION, test e collaudo)."
        )
        print(f"AVVISO: {msg}")
        return msg
    except Exception:
        return None

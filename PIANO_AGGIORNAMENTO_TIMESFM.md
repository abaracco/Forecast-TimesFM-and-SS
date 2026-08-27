# Piano di aggiornamento TimesFM 2.0.2 + performance + controllo versioni

> **File temporaneo di lavoro.** Va cancellato dal repo a collaudo superato (§ 11).
> Prima di cancellarlo, il registro decisioni (§ 12) va travasato in `CLAUDE.md`.
>
> **Versione piano: 5** — 2026-08-27
> *Dieci revisioni indipendenti (due per ciascuna delle versioni 1-5). Le correzioni rispetto
> alla v4 sono in § 13. La decima passata si è chiusa con verdetto OK da entrambi i revisori:
> nessun rilievo bloccante, il piano può partire dalla Fase 0.*

---

## 1. Contesto e diagnosi

Analisi svolta il 2026-08-27 sul repo `Forecast-TimesFM-and-SS` @ `3f48d3c` (v1.5.1)
e sull'upstream `google-research/timesfm`.

### 1.1 Problemi rilevati

| # | Problema | Gravità |
|---|---|---|
| P1 | `per_core_batch_size` non impostato → default **1** (`timesfm/src/timesfm/configs.py:55`) → `forecast()` processa **una serie alla volta**. Misurato: 195 ms/serie vs 5 ms/serie con batch 32 | **Alta** |
| P2 | Clone TimesFM **non pinnato**: `model.py:62-67` clona `master` senza tag e non aggiorna mai una cartella esistente | **Alta** |
| P3 | Deriva Colab/locale: Colab riclona `master` a ogni sessione (oggi `3dae50b`, **2 commit** avanti a `v2.0.2`), locale fermo a `a3beaa8` dell'11 mar 2026 | **Alta** |
| P4 | Loader fragile: il glob `**/*pytorch*.py` (`model.py:77`) non matcha nulla; la classe è scelta per introspezione e funziona solo perché la classe interna espone `forecast_naive` e non `forecast` | **Media** |
| P5 | La cella 6 installa `einops` (zero occorrenze in `src/` a `v2.0.2`) e **non** `safetensors`, importato a `timesfm_2p5_torch.py:26`. Funziona solo perché Colab lo ha preinstallato. *(In `requirements*.txt` `safetensors` c'è già.)* | **Media** |
| P6 | Pesi HuggingFace non pinnati: `from_pretrained` risolve sempre `main`. Se Google aggiorna i pesi, **tutti i forecast cambiano senza traccia** | **Media** |
| P7 | Nessun controllo che `forecast_lib` sia aggiornata. Tre scenari: checkout locale indietro; installazione da ZIP (autorizzata da `README.md:307`, senza `.git`); notebook vecchio in Drive + `forecast_lib` fresca clonata da `main` in Colab | **Media** |

### 1.2 Cosa NON è un problema (verificato empiricamente)

L'aggiornamento TimesFM `a3beaa8` (2.0.0) → `origin/master` **non cambia un solo numero**.
Test del 2026-08-27, stesso loader, stessa `ForecastConfig`, 3 serie fisse:

```
[OLD] point (3,24) quantili (3,24,10)   [NEW] point (3,24) quantili (3,24,10)
identici bit-a-bit: True     max|diff|: 0.0
```

Motivi verificati sul sorgente: il fix di slicing riguarda `forecast_naive`, non usato
(`forecast()` passa da `compiled_decode`); il fix `torch.compile` è inerte perché il
`config.json` su HF contiene `"torch_compile": false`, iniettato da `huggingface_hub` nel
costruttore; il fix flip-invariance AR è inerte perché `output_patch_len = 128 > horizon`
→ `num_decode_steps = 0` → `ar_outputs is None`.

**Limiti dichiarati:** il confronto è `a3beaa8` vs `origin/master`, mentre pinniamo `v2.0.2`
(i 2 commit di differenza toccano solo il fix flip-invariance AR, inerte al nostro horizon);
**i riferimenti di riga di questo piano sono presi da `a3beaa8`**, con i fatti sostanziali
riverificati a `v2.0.2` (`timesfm_2p5_torch.py` ha 43 righe di differenza fra i due);
la prova è su 3 serie sintetiche. **Il collaudo su dati reali (§ 10) è quello che vale.**

Pesi HF: fermi al 2025-10-02, revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`.
Non esiste un TimesFM 3.0.

### 1.3 Misure di riferimento

Benchmark (GPU NVIDIA, torch 2.6.0+cu124, 64 serie sintetiche, horizon 24):

| `per_core_batch_size` | tempo totale | per serie |
|---|---|---|
| 1 *(attuale)* | 12.48 s | 195 ms |
| 16 | 0.63 s | 10 ms |
| 32 | **0.31 s** | **5 ms** |

Equivalenza numerica bs=1 vs bs=32: `max|diff|` = 2.67e-5 su valori di ordine 50
(≈ **5e-7 relativo**). Vedi § 9.

**Quanto pesa la grid search** (misurato il 2026-08-27, § 10.0.1): `_grid_search_cross_origin`
su 540 SKU × 2 origini costa **3,5 s**, cioè l'**1,8 %** dei 190,9 s della cella del Modulo G.
Il resto è inferenza. Le versioni precedenti di questo piano ipotizzavano che, tolto il collo
di bottiglia del modello, quel lavoro in puro Python diventasse dominante: **non è così**.
I criteri di § 10 restano ancorati al **tempo di inferenza isolato** — è la grandezza corretta
da gatare — ma il guadagno percepito sul tempo di cella sarà quasi altrettanto grande.

*Durata reale di una run completa su file Motul: **5 m 09 s** (571 SKU), non i "10-30 minuti"
del README. Le stime di § 11 sono tarate su questo dato.*

*(`global_batch_size = per_core_batch_size * device_count`, `timesfm_2p5_torch.py:386-388`:
su multi-GPU il batch effettivo è un multiplo di `INFERENCE_BATCH_SIZE`. Per questo "Run info"
registra `model.global_batch_size`, non il parametro.)*

---

## 2. Obiettivi

1. **Performance**: eliminare il collo di bottiglia del batch size (P1).
2. **Riproducibilità**: pinnare codice TimesFM *e* pesi, con verifica (P2, P6).
3. **Aggiornamento controllato nel tempo** (P3, P7): allineamento automatico e verificato del
   clone + avvisi che non aggiornano mai da soli.
4. **Robustezza del loader**: via glob e introspezione (P4).
5. **Igiene dipendenze** (P5).
6. **Collaudo**: test automatici + regressione end-to-end su dati reali.

### 2.1 Non-obiettivi

- Modificare `metrics.py`, `calibration.py`, `inventory.py`, `rounding.py`,
  `preprocessing.py`. **Devono risultare non toccati nel diff finale.**
- Fine-tuning LoRA / covariate XReg; quantili nativi al posto della grid search;
  aggiornamento dei pesi; ottimizzazione del codice Python della grid search
  (va **misurata** e registrata); passaggio al pacchetto pip `timesfm` (§ 3.2).
- **Correggere la scelta di `q` sugli SKU con accuratezza nulla** (§ 9.1, nota): comportamento
  preesistente che il collaudo renderà visibile. Va registrato, non risolto qui.

---

## 3. Decisione architetturale

### 3.1 Scelta adottata: clone git pinnato su tag, sparse e verificato

Il loader manuale resta, ma il codice TimesFM è congelato su `v2.0.2` → P2, P3 chiusi, e P4
diventa innocuo. Si eliminano comunque glob e introspezione (§ 5.2.2) → P4 chiuso in senso stretto.

**Il clone deve essere `--filter=blob:none --sparse` + `sparse-checkout set src`**: non è
un'ottimizzazione, è l'unico modo verificato di ottenere un working tree stabilmente pulito
(§ 5.0). Verificato anche che i soli file sotto `src/` bastano al loader (import relativi
inclusi).

### 3.2 Alternativa valutata e scartata: `pip install timesfm[torch]==2.0.2`

Fattibile, toglierebbe ~80 righe. **Scartata**: non chiude nulla che il clone pinnato non
chiuda; sostituisce l'intero caricamento del modello per un problema di gravità Media;
richiederebbe una prova **pip-vs-clone** che né T4 né T5 forniscono; aggiunge in Colab una
risoluzione di dipendenze fuori dal nostro controllo. Registrata in `CLAUDE.md` come
opportunità futura.

---

## 4. Nuovi parametri (Modulo A, cella 0)

Nuovo blocco, subito dopo `MODALITA' DI ESECUZIONE`. **Va applicato per primo** (Fase 1.0).

```python
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MODELLO TIMESFM — versioni, pesi, performance                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Versione TimesFM, SENZA la "v" iniziale. Il tag git corrispondente e'
# f"v{TIMESFM_VERSION}": la "v" si aggiunge in un solo punto, dentro
# versioning.timesfm_tag().
# NB: "2.0.2" e' l'ultimo tag al 2026-08-27. La Fase 0.0 impone di
# ricontrollare all'inizio dell'implementazione e pinnare alla piu' recente
# versione collaudabile, cosi' il collaudo non nasce gia' vecchio.
TIMESFM_VERSION  = "2.0.2"
TIMESFM_REPO_URL = "https://github.com/google-research/timesfm.git"

# Se True (default) un pin non verificabile BLOCCA l'esecuzione. False fa
# proseguire con un avviso: da usare solo consapevolmente (es. GitHub
# irraggiungibile e necessita' di girare comunque). Un run non pinnato viene
# segnalato a fine esecuzione e registrato nel foglio "Run info", che in quel
# caso viene emesso anche con EXPORT_AUDIT = False.
TIMESFM_PIN_STRICT = True

# Pesi del modello su HuggingFace. La revision e' PINNATA di proposito:
# senza pin HuggingFace risolve "main" e un aggiornamento dei pesi da parte
# di Google cambierebbe tutti i forecast senza lasciare traccia.
# NB: e' pinnata, quindi non si aggiorna da sola. Il runbook "Aggiornare
# TimesFM" (README) prevede di rivalutarla a ogni cambio di TIMESFM_VERSION.
TIMESFM_MODEL_ID       = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"

# Serie inviate insieme al modello a ogni passata, per dispositivo.
# Default TimesFM = 1, cioe' una serie alla volta. Alzarlo accelera
# l'inferenza di ~40x su GPU. In caso di memoria esaurita il codice degrada
# da solo lungo la scala [N, N//4, 1]; per evitarlo, abbassare qui.
INFERENCE_BATCH_SIZE = 32

# Versione di forecast_lib attesa da questo notebook. Deve corrispondere a
# forecast_lib.__version__.
EXPECTED_FORECAST_LIB_VERSION = "1.6.0"

# Branch del repo clonato in Colab. Va tenuto su "main" in produzione;
# si cambia SOLO per collaudare un branch di lavoro (test T5).
REPO_BRANCH = "main"

# Verifica all'avvio se esistono versioni piu' recenti (TimesFM e
# forecast_lib) e stampa un avviso. Non aggiorna mai nulla da solo.
# Attivo anche in Colab: e' li' che il pin rischia di invecchiare inosservato.
CHECK_FOR_UPDATES = True

# Artefatti aggiuntivi: un SECONDO foglio "Run info" nel file Excel (la
# tabella dati resta sempre il primo foglio) e due CSV di audit accanto
# all'output. Necessari al collaudo e per ricostruire a posteriori un run.
EXPORT_AUDIT = True
```

**Fonti di verità**: `TIMESFM_VERSION` è l'unica copia del pin (non finisce nei
`requirements*.txt`). Il prefisso `v` si aggiunge solo dentro `versioning.timesfm_tag()`.

`forecast_lib/__init__.py` dichiara oggi `__version__ = "1.0.0"`, stantio (repo a v1.5.1, con
tag git da `v1.0` a `v1.5.1`). Va portato a `"1.6.0"`, allineato al tag di rilascio (item in DoD).

---

## 5. Fasi di implementazione

### Fase 0 — Scelta della versione e messa in sicurezza del clone locale *(20 min)*

> **Non chiude P3**, che si chiude in Fase 2: fino ad allora Colab clona ancora `master`.

#### 0.0 Verificare qual è l'ultima versione **al momento dell'implementazione**

`TIMESFM_VERSION = "2.0.2"` è l'ultimo tag esistente **al 2026-08-27**. Se
l'implementazione parte più tardi, pinnare a una versione già vecchia significherebbe fare
tutto il collaudo su un artefatto superato e doverlo rifare subito dopo. Quindi il primo passo
è ricontrollare, e **pinnare alla versione più recente che il collaudo può coprire**:

- [x] `git ls-remote --tags https://github.com/google-research/timesfm.git` → ultimo tag `vX.Y.Z`.
      **Eseguito il 2026-08-27: ultimo tag `v2.0.2` (`4a6c5cda`) — invariato.**
- [x] Applicare questa regola di decisione: → **riga 1 (`v2.0.2` invariato): si procede come scritto.**

| Ultimo tag trovato | Cosa fare |
|---|---|
| `v2.0.2` (invariato) | Procedere come scritto. L'analisi di § 1.2 vale così com'è |
| Nuova **patch** `v2.0.z` | **Pinnare alla nuova.** Rifare la verifica di equivalenza di § 1.2 (mezz'ora: worktree sul nuovo tag, stesso loader, stesse 3 serie fisse, confronto bit-a-bit) e annotarne l'esito in § 12 |
| Nuova **minor/major** `v2.1+` / `v3.x` | **Fermarsi e valutare con l'utente.** L'analisi di § 1.2 dimostra l'inerzia dei cambiamenti *fino a `master` di agosto 2026*: su una minor nuova non vale più. Vanno rifatti: verifica di equivalenza, controllo che l'API (`ForecastConfig`, `from_pretrained`, `compile`, `forecast`) non sia cambiata, e controllo che non sia uscito un **nuovo checkpoint** su HuggingFace — nel qual caso cambia anche `TIMESFM_MODEL_REVISION` e il collaudo diventa un confronto fra modelli diversi, non fra batch size |

- [x] Ricontrollare anche la revision dei pesi:
      `huggingface_hub.HfApi().model_info("google/timesfm-2.5-200m-pytorch").sha`.
      **Eseguito il 2026-08-27 con `git ls-remote https://huggingface.co/google/timesfm-2.5-200m-pytorch`
      (`huggingface_hub` non installato in locale): `main` = `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`,
      identica al pin. Nessun nuovo checkpoint.**
      Se diversa da `1d952420…`, **non** adottarla in automatico: pinnare quella attuale e
      valutare la nuova a parte, perché cambierebbe tutti i numeri e non è ciò che T4 misura.
- [x] Aggiornare `TIMESFM_VERSION` (e, se serve, `TIMESFM_MODEL_REVISION`) nel § 4 e nel
      Modulo A **prima** di iniziare la Fase 1, così tutto il collaudo gira sulla versione
      definitiva. Registrare la scelta in § 12. **Nessuna modifica necessaria: entrambi i valori
      già scritti nel § 4 sono quelli attuali. Il Modulo A del notebook non contiene ancora il
      blocco — lo introduce la Fase 1.0.**

> **Il principio:** si pinna sempre alla versione più recente *che siamo in grado di
> collaudare adesso*. Il pin non serve a restare indietro, serve a non muoversi da soli.

#### 0.1 Messa in sicurezza del clone locale

**Perché non `git checkout`, e perché non basta `GIT_LFS_SKIP_SMUDGE`.** Il repo TimesFM ha un
`.gitattributes` con filtri LFS su `timesfm-forecasting/**/*.png|gif`, ma quei 4 file sono
committati come blob reali, non come pointer. Verificato sul campo (git 2.55, git-lfs 3.7.1,
**15 cloni ripetuti**):

| variante | tree pulito? |
|---|---|
| `git checkout` fra tag | **no** — 4 file sporchi, e il cambio di tag successivo aborta |
| `git clone` semplice | **no** |
| `GIT_LFS_SKIP_SMUDGE=1 git clone` | **no** — 14 su 15 sporchi |
| `-c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false` | sì (3/3) |
| **`--filter=blob:none --sparse` + `sparse-checkout set src`** | **sì (4/4), stabile anche dopo `touch`** |

`GIT_LFS_SKIP_SMUDGE` disattiva solo lo *smudge*; è il filtro **clean**, eseguito da
`git status`, a marcare i file come modificati. La variante sparse è robusta perché quei file
**non esistono** nel working tree. In più pesa **338 KB** invece di 6.9-8.4 MB, in ~1,5 s.

- [x] Cancellare `./timesfm/` (gitignorata, `.gitignore:42`, rigenerabile). Su Windows i file
      `.git/objects/pack/*.idx|.pack` sono **read-only**: `shutil.rmtree` solleva
      `PermissionError [WinError 5]` (verificato). Serve un handler `onexc`/`onerror` con
      `os.chmod(path, stat.S_IWRITE)`. **Fase 0 eseguita da PowerShell con
      `Remove-Item -Recurse -Force`, che gestisce già i read-only; l'handler resta necessario
      per il codice Python della Fase 2.**
- [x] Riclonare (**PowerShell**, shell primaria di questa macchina):
      ```powershell
      git clone --depth 1 --branch v2.0.2 --filter=blob:none --sparse `
                https://github.com/google-research/timesfm.git .\timesfm
      git -C .\timesfm sparse-checkout set src
      ```
- [x] **Criteri di uscita** (entrambi): **verificati il 2026-08-27.**
      `git -C timesfm rev-parse HEAD` == `git -C timesfm rev-parse refs/tags/v2.0.2^{commit}`
      → entrambi `4a6c5cdae7ac73a450843e035fbfc3ffa08e9caf` ✅
      **e** `git -C timesfm status --porcelain` **vuoto** ✅
      Clone risultante: **227 KB** (era 9,7 MB a `a3beaa8`), `src/` presente.

> *Perché non `git describe`:* su `a3beaa8` restituisce `v1.2.6-117-ga3beaa8`. Si confrontano
> hash di commit.
> *Nota:* `--filter=blob:none` produce un repo *promisor*: operazioni future che richiedano
> blob mancanti esigono rete. Le verifiche del piano girano tutte offline (verificato).

---

### Fase 1 — Performance e robustezza dell'inferenza *(~6 h)*

#### 1.0 Modulo A (prerequisito)
- [x] Inserire il blocco parametri di § 4. Senza, la Fase 1.6 rompe il notebook.

#### 1.1 `forecast_lib/model.py` — `setup_timesfm()`
- [x] Nuovi parametri keyword-only: `batch_size=32`, `model_revision=None`.
      *(`model_id` esiste già, `model.py:35`. `expected_version` arriva in Fase 2, dove viene
      usato: aggiungerlo qui lo lascerebbe accettato e ignorato.)*
- [x] `per_core_batch_size=batch_size` in `ForecastConfig(...)`; `revision=model_revision` in
      `from_pretrained(...)`.
- [x] Rimuovere il ramo morto `model.compile()` senza argomenti (`model.py:138`): la classe
      concreta sovrascrive il default della base con `compile(self, forecast_config, **kwargs)`
      → `TypeError`. Se `ForecastConfig` non si trova → errore esplicito.
- [x] Rimuovere `os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")` (`model.py:54`):
      con `huggingface_hub` 1.x (nel `.venv`: 1.16.1) è inerte e produce `FutureWarning`.
- [x] **`import torch` resta lazy** (oggi `model.py:141`): la cella 1 importa `forecast_lib`
      **prima** del `!pip install torch` della cella 6.

#### 1.2 Stato allegato al modello (niente cambio di firma)

`setup_timesfm` continua a restituire **solo** il modello: celle 6, 7 e 8 fanno
`model = fl_model.setup_timesfm(...)` e passano `model` a `run_backtest` e
`forecast_all_skus_point`. Lo stato aggiuntivo viaggia come attributi `fl_`:

- [x] `model.fl_recompile(batch_size)` — closure che **riusa la `ForecastConfig` originale**,
      sostituendo solo `per_core_batch_size`. *Non* `dataclasses.replace(model.forecast_config, …)`:
      dopo `compile()` quella config ha `max_horizon = 128`, non 24.
      **`global_batch_size` è impostato solo dentro `compile()`**, quindi ogni cambio di batch
      (incluso il degrado di § 1.3) **deve** passare da qui.
- [x] `fl_batch_size`, `fl_degraded`, `fl_degraded_after_inference`, `fl_device`,
      `fl_model_revision`, `fl_inference_seconds`, `fl_pin_verified`.
- [x] **Risoluzione della revision dei pesi** (`fl_model_revision`): né `from_pretrained` né
      `_from_pretrained` espongono il path dello snapshot. Va risolta a parte:
      `hf_hub_download(model_id, "config.json", revision=...)` prima del caricamento,
      estraendo l'hash dal path `snapshots/<hash>/`.

> **Decisione di design (esplicita):** si iniettano attributi su un oggetto di terza parte
> (lecito: `TimesFM_2p5` è una classe normale, senza `__slots__` e non dataclass — verificato).
> L'alternativa pulita è una classe wrapper, più invasiva perché `backtest.py:208` chiama
> `model.forecast(...)` direttamente. Prefisso `fl_` contro le collisioni.

#### 1.3 `forecast_batch_with_fallback()` — helper condiviso

**Contratto (nel docstring):**
`forecast_batch_with_fallback(model, inputs, horizon, count_time=True) -> (list[np.ndarray | None], dict[int, str])`
— una previsione per input, nell'ordine dato, `None` per i falliti, più `indice → errore`.
I chiamanti mantengono la semantica attuale: `forecast_all_skus_point` costruisce
`(results, errors)` per SKU; `run_backtest` scarta gli SKU con `base_fc is None`.

Livelli:
- [x] 1. batch pieno al `INFERENCE_BATCH_SIZE` corrente;
- [x] 2. su OOM (`torch.cuda.OutOfMemoryError` o `RuntimeError` con `out of memory`):
      degrado lungo la scala **`[N, max(1, N//4), 1]`** derivata da `INFERENCE_BATCH_SIZE`
      (con `N = 32` dà 32 → 8 → 1), un livello alla volta, via `fl_recompile`.
      *Motivo: le due chiamate reali passano tutte le serie insieme (`model.py:203`,
      `backtest.py:208`) e TimesFM le spezza internamente. Cadere direttamente a 1 butta via
      ~40x quando spesso basta un livello intermedio. La scala è derivata, non costante,
      perché `INFERENCE_BATCH_SIZE` è configurabile.*
- [x] 3. su qualunque altro fallimento: **prima degradare a batch 1**, poi loop per-input.
      *Motivo: `forecast()` padda ogni chiamata a `global_batch_size` e con
      `force_flip_invariance=True` il decode gira due volte: un loop per-input al batch 32
      farebbe 32 serie di calcolo per ogni risultato utile.*
- [x] Log distinti per i tre casi.
- [x] Accumulare in `model.fl_inference_seconds` il tempo delle chiamate a `model.forecast`,
      **escluso lo smoke test** (`count_time=False`): a batch 32 lo smoke test padda 1 serie a
      32 e inquinerebbe l'unica metrica di performance del collaudo.

**Quattro requisiti obbligatori, ciascuno per un difetto verificato:**

- [x] **(a) Passare sempre una copia della lista.** `timesfm_2p5_base.py:166-168` fa
      `inputs += [np.array([0.0]*3)] * (...)`: **muta in place la lista del chiamante**.
      La prima chiamata resta corretta (`return output_points[:num_inputs]` usa `num_inputs`
      catturato prima del padding), ma un **retry sulla stessa lista** restituisce righe
      fantasma (verificato: 5 elementi → 32, retry → 32 righe).
      → `model.forecast(horizon=..., inputs=list(inputs))`.
- [x] **(b) Liberare la memoria CUDA prima del retry.** Ritentare dentro l'`except` mantiene
      vivo `e.__traceback__` e con esso i frame del forward andato in OOM. → uscire
      dall'`except`, poi `gc.collect()` e `torch.cuda.empty_cache()`.
- [x] **(c) Degrado permanente per il run**, con `fl_degraded = True`.
      > **Il degrado NON rende il run coerente.** `run_backtest` chiama il modello una volta
      > per origine (`backtest.py:208`) e condivide l'istanza con la cella 8: se l'OOM scatta
      > all'origine 2, l'origine 1 è già stata calcolata al batch precedente. Il degrado evita
      > solo di alternare avanti e indietro. **Un run degradato a metà pipeline non è
      > consegnabile** e va rifatto con `INFERENCE_BATCH_SIZE` più basso.
- [x] **(d) Distinguere il degrado nello smoke test.** Se l'OOM scatta lì (§ 1.4), avviene
      **prima di qualunque inferenza reale**: il run gira uniformemente al batch degradato ed
      è **pienamente utilizzabile**. `fl_degraded_after_inference` resta `False` e l'avviso
      "rifare il run" **non** va emesso.

#### 1.4 Smoke test
- [x] Gira **dopo** la compilazione al batch di produzione e **attraverso**
      `forecast_batch_with_fallback(..., count_time=False)`. Con batch 32 e una sola serie
      attiva subito il ramo di padding: è un canary del percorso nuovo.
- [x] Verifica che l'output sia sensato (shape attesa, valori finiti), non solo che non sollevi.
- [x] **Solleva** se fallisce dopo tutti i livelli di fallback. Cambio di comportamento voluto,
      da annotare nel README.

#### 1.5 `forecast_lib/backtest.py`
- [x] Sostituire il `try/except` di righe 207-227 con `forecast_batch_with_fallback`,
      semantica a valle invariata (SKU senza forecast scartati alla riga 230).
- [x] **Aggiungere `BestQuantileRaw` e `BestAccuracyRaw`** al `results_list`
      (`backtest.py:351-356`). Oggi `_apply_shrinkage` sovrascrive in place **entrambi**:
      `r["BestQuantile"] = q_shrunk` (`:380`) e `r["BestAccuracy"] = np.mean(accs)` (`:396`,
      cioè l'accuratezza **al `q` shrinkato**, non il massimo sulla griglia). Senza i valori
      grezzi non è possibile distinguere un flip individuale dallo spostamento della mediana,
      né leggere il KPI come "massimo su griglia".
- [x] **Esporre `q_global`** (oggi solo `print`ato, `backtest.py:104`) **senza cambiare la
      firma** — `run_backtest` termina con `return pd.DataFrame(results_list)` e la cella 7 fa
      `df_backtest_results = fl_backtest.run_backtest(...)`: una tupla romperebbe il call site.
      → `df.attrs["q_global"]` **e** una colonna costante nel CSV di audit (`df.attrs` non
      sopravvive a `to_csv`, e la diagnostica di § 9 lo richiede).
      Casi in cui **non esiste**, da gestire: `SHRINKAGE_ENABLED = False`, `results_list`
      vuoto, `RUN_BACKTEST = False`.
- [x] Conteggio degli SKU esclusi dal backtest e degli SKU con **`BestAccuracyRaw == 0`**
      (§ 9.1: lì il `q` è un artefatto dell'ordine di iterazione, non una scelta del modello).
- [x] Nessuna modifica a grid search, shrinkage o calibrazione.

#### 1.6 `forecast_lib/export.py`, notebook celle 6, 7, 12
- [x] Cella 6: `batch_size=INFERENCE_BATCH_SIZE`, `model_id=TIMESFM_MODEL_ID`,
      `model_revision=TIMESFM_MODEL_REVISION`. Firma di ritorno invariata (§ 1.2).
- [x] `save_excel(df, path)` → `save_excel(df, path, run_info=None)`. Oggi è
      `df.to_excel(path, index=False)` (`export.py:105-107`), foglio unico: per un secondo
      foglio serve `pd.ExcelWriter`. **Vincolo: la tabella dati resta il PRIMO foglio** —
      questo stesso progetto legge l'input con `list(all_sheets.keys())[0]` (cella 2).
- [x] Foglio "Run info" — emesso se `EXPORT_AUDIT` **oppure** se `fl_pin_verified is False`
      (un run non pinnato deve lasciare traccia comunque): data/ora,
      `forecast_lib.__version__`, `TIMESFM_VERSION` risolta, `fl_pin_verified`, revision pesi
      risolta, **device** — con `torch.cuda.get_device_name()`, non solo `"cuda"/"cpu"`:
      Colab riassegna T4/L4/A100 fra sessioni, e senza il modello esatto un fallimento spurio
      di G3/G4 in T5 non sarebbe diagnosticabile a posteriori —, `model.global_batch_size`
      effettivo, `fl_degraded` / `fl_degraded_after_inference`, `fl_inference_seconds`,
      **KPI Motul aggregato** (`Σ(BestAccuracy×TotalWeight)/Σ(TotalWeight)`, già calcolato
      dalla cella 7 — ma **solo** dentro il ramo `RUN_BACKTEST = True` e con
      `df_backtest_results` non vuoto: fuori da lì il riferimento diretto è un `NameError`,
      quindi va gestito a `None` come `q_global`),
      `BUSINESS_ADJUSTMENT_FACTOR`, `RUN_BACKTEST`, `N_BACKTEST_ORIGINS`, `SHRINKAGE_ENABLED`,
      `q_global`, `ROUNDING_MODE`, n. SKU esclusi dal backtest, n. SKU con `BestAccuracyRaw == 0`.
- [x] **Avviso di fine run**, indipendente da `EXPORT_AUDIT`, se `fl_pin_verified is False`
      o `fl_degraded_after_inference is True`. La cella 6 scorre via nel log; l'ultima cella no.
- [x] Due CSV di audit in `OUTPUT_DIR` se `EXPORT_AUDIT`. Naming coerente con la cella 12 —
      `OUTPUT_SUFFIX` **inizia con uno spazio**, quindi comporre come
      `f"{OUTPUT_FILE_BASE} backtest{OUTPUT_SUFFIX}.csv"`: `df_backtest_results`
      (`BestQuantile`, `BestQuantileRaw`, `BestAccuracy`, `BestAccuracyRaw`, `TotalWeight`,
      `q_global`) e `fc_errors`.
- [x] **Cella 12**: in Colab `files.download()` anche per i due CSV (`/content/output/` è
      effimero: senza download T5 non può verificare i criteri di § 9).

---

### Fase 2 — Versioning e controllo aggiornamenti *(~5 h)*

#### 2.1 `forecast_lib/versioning.py` (nuovo)

I **check informativi** non sollevano mai, hanno timeout ≤ 5 s e girano con
`GIT_TERMINAL_PROMPT=0`. La **verifica del pin** è bloccante per default.

> **Nota sull'ambiente dei `subprocess`**: usare l'**ambiente ereditato** (eventualmente con
> `GIT_TERMINAL_PROMPT=0` aggiunto), **mai `env={}`**. Verificato: `env={}` fa fallire il
> clone HTTPS su Windows con exit 128 (`Could not resolve host`).

##### 2.1.1 `ensure_timesfm_checkout(timesfm_dir, repo_url, tag, strict=True) -> dict`

1. cartella assente → clone sparse (§ Fase 0);
2. cartella presente ma non repo git → errore esplicito con istruzione (cancellarla);
3. cartella presente: **verificare `git remote get-url origin == repo_url`**. Se diverso →
   **sollevare, mai cancellare**. `timesfm_dir` è `"./timesfm"`, relativo alla CWD, e
   `CLAUDE.md` avverte che in JupyterLab/VS Code la CWD può non essere la root del repo.
   *(La guardia del punto 2 non basta: un fork **è** un repo git.)*
4. cartella corretta, `HEAD == refs/tags/<tag>^{commit}` e `status --porcelain` vuoto → nulla;
5. altrimenti → **clonare in una directory temporanea e fare lo swap solo a clone riuscito**.
   Mai cancellare prima: se il clone fallisce (rete giù — lo scenario per cui esiste
   `TIMESFM_PIN_STRICT = False`), la cartella esistente deve restare utilizzabile.
   Per la rimozione della vecchia cartella, `shutil.rmtree` con handler `onexc`/`onerror` che
   fa `os.chmod(path, stat.S_IWRITE)`: su Windows i `.pack`/`.idx` sono read-only.
   **La directory temporanea va creata accanto a `timesfm_dir`**
   (`tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(timesfm_dir)))`), **non in `%TEMP%`**:
   uno swap cross-volume fallisce — verificato con repo su `D:` e `%TEMP%` su `C:`,
   `os.replace` → `OSError [WinError 17]`, e `shutil.move` copia correttamente ma poi solleva
   `PermissionError [WinError 5]` sui `.idx` read-only del sorgente, perché il suo `rmtree`
   interno non ha handler.
6. **verifica finale**: `HEAD == refs/tags/<tag>^{commit}` **e** `status --porcelain` vuoto.
   Con `strict=True` → **solleva**; con `strict=False` → avviso rumoroso e
   `pin_verified=False`, propagato a "Run info" e all'avviso di fine run (§ 1.6).
- [x] Fallback di clone se `--filter=blob:none` non è supportato:
      `-c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false`
      (verificato pulito 3/3). Il criterio di accettazione resta il punto 6.

##### 2.1.2 Check informativi (non bloccanti)
- [x] `timesfm_tag(version)` — **unico punto** in cui si aggiunge il prefisso `v`.
- [x] `latest_timesfm_tag(repo_url)`, `latest_lib_version(repo_url)` — `git ls-remote --tags`.
      `latest_lib_version` **chiude il caso ZIP**, che `EXPECTED_FORECAST_LIB_VERSION` da solo
      non copre (uno ZIP è coerente al proprio interno). *(Niente fallback HTTP: `git` è un
      requisito duro della pipeline — senza, `ensure_timesfm_checkout` non può clonare — quindi
      sarebbe codice morto. Va però documentato nel README, che oggi autorizza lo ZIP senza dirlo.)*
- [x] `compare_versions(a, b)` — parsing numerico dei segmenti (`2.0.10 > 2.0.9`), tollera il
      prefisso `v`, **gestisce numero di segmenti variabile** (`v1.2` accanto a `v1.4.1`: i
      tag reali del repo sono così), non solleva su input malformato. Il parsing di
      `ls-remote` deve **deduplicare le righe `refs/tags/<t>^{}`** (tag misti annotati/leggeri).
- [x] `local_repo_status(repo_path, expected_remote)` — `behind`, `ahead`, `dirty`, `branch`.
      **Prima di qualunque `fetch`, verificare `git remote get-url origin == expected_remote`.**
      Mai `pull`, `merge`, `checkout`.
- [x] `check_library_version(actual, expected) -> str | None` — confronto puro, **due
      messaggi distinti**: `actual < expected` → *"forecast_lib è più vecchia di quanto il
      notebook si aspetti: aggiorna il codice"*; `actual > expected` → *"il notebook è più
      vecchio del codice: riscarica il notebook aggiornato"*.

##### 2.1.3 Due orchestratori distinti
- [x] `check_project_updates(*, colab, enabled, repo_path, repo_url, lib_version,
      expected_lib_version)` → **cella 1**. `check_library_version` gira sempre (nessun I/O);
      `latest_lib_version` se `enabled`, **anche in Colab**; `local_repo_status` solo se non-Colab.
- [x] `check_timesfm_update(*, enabled, repo_url, current_version)` → **cella 6**, dopo
      `ensure_timesfm_checkout`. **Attivo anche in Colab**: è lì che il pin rischia di
      invecchiare inosservato, essendo Colab la modalità principale.

Le guardie stanno **dentro** queste funzioni, non nel notebook, così sono testabili con pytest.
In tutti i casi degradati: **`None` + avviso, mai eccezione** — unica eccezione voluta,
`ensure_timesfm_checkout` in modalità strict.

#### 2.2 `forecast_lib/model.py` — loader esplicito
- [x] `ensure_timesfm_checkout(...)` al posto del clone condizionale; nuovi parametri
      `expected_version`, `repo_url`, `pin_strict`.
- [x] Path diretto `src/timesfm/timesfm_2p5/timesfm_2p5_torch.py` al posto del glob; errore
      esplicito se assente.
- [x] `getattr(torch_mod, "TimesFM_2p5_200M_torch")` al posto dell'introspezione; errore
      esplicito se assente. *(Verificato presente a `v2.0.2`, `timesfm_2p5_torch.py:259`.)*
- [x] Aggiornare il docstring del modulo.

#### 2.3 Notebook — cella 1
- [x] Aggiungere `versioning` alla `from forecast_lib import (...)` — senza,
      `versioning.check_project_updates(...)` è un `NameError`. E alla mappa dei moduli nel
      docstring di `__init__.py`. Bump `__version__ = "1.6.0"`.
- [x] **Allineamento del branch in Colab.** Il clone è `--depth 1`, che implica
      `--single-branch`: `remote.origin.fetch` è `+refs/heads/main:refs/remotes/origin/main`,
      quindi né `git pull` né `fetch`+`checkout` semplici funzionano per un altro branch
      (verificato: `pathspec did not match`; e dopo un `checkout -B … FETCH_HEAD` il branch
      **non ha upstream**, quindi il `git pull` successivo muore con
      `refusing to merge unrelated histories`). Eseguire **sempre** questa sequenza, che è
      idempotente e non dipende dallo stato precedente:
      ```
      # solo se la refspec non e' gia' presente, per non duplicarla
      git -C <path> config --get-all remote.origin.fetch | grep -q <REPO_BRANCH> \
        || git -C <path> remote set-branches --add origin <REPO_BRANCH>
      git -C <path> fetch --depth 1 origin <REPO_BRANCH>
      git -C <path> checkout -B <REPO_BRANCH> FETCH_HEAD
      git -C <path> branch --set-upstream-to=origin/<REPO_BRANCH> <REPO_BRANCH>
      ```
      **Non** usare il ramo alternativo "se il branch è già quello giusto, basta `git pull`":
      è proprio quello che fallisce in silenzio e farebbe collaudare a T5 il codice vecchio.
- [x] `fetch`/`checkout` **non bloccanti** (niente `check=True`) ma con **esito verificato e
      stampato**: se falliscono, dirlo a voce alta, perché il notebook proseguirà con il
      codice della sessione precedente. Il `clone` iniziale resta bloccante, con messaggio
      esplicito se il branch non esiste.
- [x] Stampare l'avviso `REPO_BRANCH != "main"` **prima** del clone.
- [x] **Documentare che serve il riavvio del kernel**: dopo un aggiornamento i moduli già
      importati restano in `sys.modules`.
- [x] Chiamare `versioning.check_project_updates(...)`.

---

### Fase 3 — Igiene dipendenze *(~20 min)*

- [x] Cella 6: `!pip install -q safetensors huggingface_hub torch` (aggiunto `safetensors`,
      rimosso `einops`).
- [x] Rimuovere `einops` da `requirements.txt` e `requirements-nvidia.txt`.
- [x] Non toccare i pin di `torch` (2.6.0 compatibile: TimesFM chiede `torch>=2.0.0`).

---

### Fase 4 — Test automatici *(~7 h)*

- [x] `pytest.ini`: `markers = slow: richiede rete o il download del modello TimesFM` e
      `addopts = -m "not slow"`. *(Verificato con pytest 9.1.1: `-m slow` da riga di comando
      sovrascrive l'ini. `pytest.ini` oggi non ha `addopts`, nessun conflitto.)*
- [x] Scrivere `tests/test_versioning.py`, `tests/test_versioning_integration.py`,
      `tests/test_model_config.py`, `tests/test_model_integration.py`,
      `tests/tools/compare_forecast_outputs.py`, e i casi su `save_excel` in `tests/test_export.py`.

---

### Fase 5 — Documentazione *(~1 h 30)*

- [x] `CLAUDE.md`: "Key Configuration" con tutti i parametri di § 4; "Project Layout" con
      `versioning.py`; Modulo F aggiornato; nuove "Important Design Decisions": *pin di
      versione e pesi con verifica*, *clone sparse per i filtri LFS*, *degrado su OOM e run
      non consegnabile*, *pip valutato e scartato*, *scelta di `q` sugli SKU ad accuratezza
      nulla* (§ 9.1); travaso del registro decisioni (§ 12).
- [x] `README.md` — runbook **"Aggiornare TimesFM"**: cambiare `TIMESFM_VERSION` →
      **rivalutare `TIMESFM_MODEL_REVISION`** (ora che è pinnata non si aggiorna da sola:
      codice nuovo con pesi vecchi è uno scenario reale) → `pytest` → `pytest -m slow` →
      T4 → confronto → aggiornare la doc.
- [x] `README.md` — correggere riga 39 (Modulo F) e riga 407 (togliere `einops`, aggiungere
      `safetensors`). *(Riga 414 resta vera: TimesFM da GitHub, non da pip.)*
- [x] `README.md` — documentare: `INFERENCE_BATCH_SIZE` e la scala di degrado; i due flag di
      degrado; il foglio "Run info" e i CSV; lo smoke test ora bloccante; che il pin della
      revision HF **migliora** l'uso offline *(nota: il clone TimesFM è invece promisor, quindi
      operazioni git future su blob mancanti richiedono rete)*; che **`git` serve anche a chi
      installa da ZIP** (riga 307).

---

## 6. File toccati

| File | Tipo | Fase |
|---|---|---|
| `Forecast_TimesFM_and_SS.ipynb` (celle 0, 1, 6, 7, 12) | modifica | 1, 2, 3 |
| `forecast_lib/model.py` | modifica sostanziale | 1, 2 |
| `forecast_lib/backtest.py` | modifica localizzata | 1 |
| `forecast_lib/export.py` | `save_excel` + foglio "Run info" | 1 |
| `forecast_lib/versioning.py` | **nuovo** | 2 |
| `forecast_lib/__init__.py` | bump versione + mappa moduli | 2 |
| `requirements.txt`, `requirements-nvidia.txt` | modifica | 3 |
| `pytest.ini` | marker `slow` | 4 |
| `tests/test_versioning.py`, `test_versioning_integration.py`, `test_model_config.py`, `test_model_integration.py` | **nuovi** | 4 |
| `tests/test_export.py` | casi su `save_excel` | 4 |
| `tests/tools/compare_forecast_outputs.py` | **nuovo** | 4 |
| `CLAUDE.md`, `README.md` | modifica | 5 |
| `PIANO_AGGIORNAMENTO_TIMESFM.md` | **da cancellare a fine lavori** | 11 |

**Non devono comparire nel diff:** `metrics.py`, `calibration.py`, `inventory.py`,
`rounding.py`, `preprocessing.py`.

---

## 7. Rischi e mitigazioni

| Rischio | Prob. | Impatto | Mitigazione |
|---|---|---|---|
| **Cambio di `q` su una quota di SKU** (§ 9) | Media | Medio | Gate non compensativi + gate "ogni scostamento grande deve essere spiegato da un flip di `q`" (§ 9.2) |
| Clone TimesFM sporco per i filtri LFS → pin non verificabile → blocco | Alta se non gestito | **Alto** | Clone sparse (verificato 4/4 pulito) + `TIMESFM_PIN_STRICT` + clone in temp con swap |
| `rmtree` fallisce su Windows sui `.pack` read-only | Certa se non gestito | Alto | Handler `onexc` con `chmod` (§ 2.1.1) |
| Cancellazione della cartella sbagliata | Bassa | **Alto** | Verifica del remote **prima** di toccare qualsiasi cosa (§ 2.1.1 punto 3) |
| Righe fantasma da padding nei retry | Alta se non gestito | Alto | Copia della lista (§ 1.3a); T1.c/T1.f |
| OOM su GPU Colab | Bassa | Medio | Degrado `[N, N//4, 1]` + `empty_cache` + parametro abbassabile |
| Run degradato a metà pipeline scambiato per valido | Bassa | Alto | `fl_degraded_after_inference` + avviso di fine run + "Run info" |
| T5 collauda il branch sbagliato | Media se non gestito | Alto | Sequenza idempotente a 4 comandi con esito verificato (§ 2.3) |
| Run non pinnato senza traccia | Bassa | Alto | "Run info" forzato + avviso di fine run quando `pin_verified = False` |
| Foglio "Run info" rompe letture lato cliente | Bassa | Medio | Dati sempre primo foglio + toggle `EXPORT_AUDIT` |
| Check di rete lenti, falliti o bloccati su prompt | Media | Basso | Timeout 5 s, `GIT_TERMINAL_PROMPT=0`, non bloccanti |

---

## 8. Nota sui pesi HuggingFace

La revision `1d952420…` è già in cache locale ed è quella a cui punta `refs/main`. Il pin
**non** provoca ri-download né duplicazione di cache, e **migliora** il comportamento offline
(HF non deve risolvere `main`). Costo zero; unico effetto: i pesi non cambiano più da soli —
motivo per cui il runbook di § 5 li rivisita a ogni aggiornamento di TimesFM.

---

## 9. Criteri di accettazione numerica

### 9.1 Cosa può cambiare, e perché contarlo non funziona

Il cambio di batch size perturba i risultati di `ε ≈ 5e-7` relativo (§ 1.3). L'**unico**
meccanismo con cui questo diventa visibile è l'attraversamento di una soglia di
`np.round(v/pack)` (`rounding.py:40`). Tre vie di propagazione, tutte verificate:

1. **Arrotondamento diretto del forecast consegnato.** Misurato su un file di output reale
   del progetto (**576 SKU**, `output/Forecast and SS 2026 03 19 17_09.xlsx`):
   `v/pack` mediana **20**, media **109**, p90 **246**; `Σ(v/pack) ≈ 1.3e6` su 12.086 valori
   → attesa **&lt; 1 valore** che cambia di un pack sull'intero deliverable. Trascurabile.
   *(La v4 citava media 836: quello è il file dimostrativo da 30 SKU. Numeri rifatti.)*
2. **Argmax della grid search.** `best_q = max(all_acc, key=...)` (`backtest.py:328`) su una
   funzione **a gradini**. Ogni SKU subisce ~624 arrotondamenti nella griglia (12 mesi ×
   ~26 `q` × 2 origini): con `v/pack` medio 109 la probabilità che almeno uno flippi è
   **~3,4% per SKU**, e solo una frazione di questi sposta l'argmax. **Attesa: ordine
   dell'1% degli SKU con `q` diverso**, cioè ~5 SKU su 576.
   Ma quando il `q` cambia, cambia **molto**: la griglia decide prima al passo **coarse
   0.05** (`backtest.py:300`), e `scale = q/0.5`, quindi un flip coarse vale **≥ 10%** di
   volume su quello SKU. Su un plateau esatto `max()` restituisce la **prima chiave** di
   `{**coarse_acc, **fine_acc}`, cioè `q = 0.10`: il salto può essere arbitrario.
3. **Mediana dello shrinkage.** `_apply_shrinkage` calcola `q_global = np.median(q_values)`
   una volta sola, poi `q_shrunk = round(alpha*q_orig + (1-alpha)*q_global, 2)` con
   `alpha = min(1, hist_len/36)` (`backtest.py:371-381`). Un solo flip può spostare la
   mediana e cambiare `q` a **tutti** gli SKU con meno di 36 mesi di storico.
   *È il meccanismo che spaventa di più a leggerlo, ma sul file misurato è il meno rilevante:
   solo **76 SKU su 576** hanno storico < 36 mesi e valgono il **2,5%** del volume, quindi uno
   spostamento di `q_global` di un passo fine muove l'aggregato dello **0,023%** — due ordini
   di grandezza sotto G3, pur essendo tutto dello stesso segno.*

> **Caso limite preesistente, da conoscere.** Se la formula Motul restituisce 0 su tutta la
> griglia — frequente per gli SKU erratici (`accuracy_single_month` azzera per `FCST < ACT/2`
> o `> 2×ACT`; `accuracy_weighted` torna 0 se `total_w <= 0`) — `all_acc` è tutto zero e
> `max()` restituisce `q = 0.10`, primo valore di `QUANTILE_GRID`. Su quegli SKU il `q` non è
> una scelta del modello ma un artefatto dell'ordine di iterazione. **Non sono confinati in
> classe C**: ABC è Pareto sul volume, l'accuratezza nulla dipende dall'erraticità (XYZ) —
> uno SKU A/Z è il candidato tipico. Per questo il conteggio si fa su `BestAccuracyRaw == 0`,
> non sulla classe. **Correggerlo è fuori scope** (§ 2.1), ma va registrato in § 12.

**Perché non si gata su un conteggio.** Le versioni 2, 3 e 4 di questo piano ci hanno provato
tre volte: soglie sul numero di valori diversi, sulla percentuale di SKU flippati, su un cap
di volume per-SKU. Tutte e tre sono risultate, ai numeri reali, o tarate sul valore atteso
(→ ~50% di falsi allarmi) o incompatibili con l'attesa dichiarata poche righe sopra (un flip
coarse vale ≥10% e sfonda qualunque cap ragionevole). **Il conteggio dei flip non è una
misura di qualità**: entrambe le run scelgono un punto di un plateau, e quale dei due sia
"giusto" è indecidibile.

**Dove sta la garanzia, allora.** Nel criterio 1: se il refactor è bit-identico a batch 1,
l'unica variabile residua in T4.2 è il batch size, il cui effetto è **provatamente limitato**
a `ε ≈ 5e-7` e ai suoi effetti a soglia. Non può introdurre un bias sistematico né perdere
SKU. I gate di § 9.2 servono quindi a due cose: (a) verificare che l'impatto aggregato resti
piccolo, (b) intercettare qualunque differenza che **non** sia spiegabile con questo
meccanismo — che sarebbe un bug.

### 9.2 I gate

**G1 — Identità del refactor (T4.1). Il gate principale.**
Codice nuovo con `INFERENCE_BATCH_SIZE = 1` vs codice di `main`: **bit-identici** sulle
colonne forecast e sul CSV di backtest (confronto `NaN`-aware, es. `DataFrame.equals`:
`build_final_table` fa merge `how="left"`, quindi gli SKU senza forecast hanno `NaN`).
Tutte le modifiche di Fase 1-2 sono numericamente neutre a batch 1 (verificato:
`per_core_batch_size=1` == default della libreria; il ramo di padding è morto a batch 1,
quindi `list(inputs)` è neutro; `model.compile()` senza argomenti è irraggiungibile).
**Una differenza qui è un bug del refactor, non un effetto del batching.**

**G2 — Identità strutturale (T4.2).** Devono coincidere esattamente:
- `ABC`, `XYZ`, `SafetyStock`, **`LT`** *(nell'Excel si chiama così: `export.py:76` rinomina
  `LT_Final` → `LT`; non esiste alcuna colonna `ROP`)*. Verificato che
  `calculate_inventory_logic(df_history, meta_df, …)` non riceve il forecast: sono identici
  **per costruzione**, quindi è un controllo di sanità della pipeline, non del forecast;
- l'**insieme degli SKU con un risultato di backtest** e l'**insieme degli SKU falliti**
  (`fc_errors`). Non è cosmetico: uno SKU che sparisce dal backtest ricade su
  `best_q_map.get(sku, 0.5)` nel Modulo H, cioè fino a **5×** di scostamento.
  *(Il "numero di SKU in output" **non** è un criterio: `build_final_table` parte dai
  metadati con merge `how="left"`, quindi è invariante per costruzione.)*

**G3 — Impatto aggregato con segno.** Volume complessivo previsto (somma dei 24 mesi su tutti
gli SKU): scostamento **≤ 0.5%**.

**G4 — Impatto aggregato non compensativo.** `Σ|Δ volume_SKU| / Σ volume_baseline` **≤ 1%**.
Serve perché G3, avendo segno, si compensa: la classe C è il 9,8% del volume ma **399 SKU su
576** nel file misurato, e una redistribuzione ampia fra SKU C passerebbe G3 indenne.

> **Cosa aspettarsi davvero da G3 e G4** (da leggere *prima* di guardare il report di T4.2,
> altrimenti un risultato normale verrà scambiato per un bug). Gli SKU che flippano `q` non
> sono un campione casuale: `P(flip) ∝ v/pack`, che è fortemente correlato al volume. Sul
> file misurato la media di `v/pack` **pesata a volume** è 556 contro 95 non pesata: **5,9×**.
> Quindi un 1% di SKU flippati a conteggio vale ~**5%** del volume, non l'1%. Simulazione su
> quel file, assumendo `|Δ| ≈ 10%` per flip (§ 9.1 punto 2):
> **G3 attesa ≈ 0,2-0,3%** (p90 0,5%) → probabilità di scatto **~10%**;
> **G4 attesa ≈ 0,4-0,5%** (p90 0,9%) → probabilità di scatto **~5%**.
> Le soglie restano quelle giuste — uno scatto porta a **un'indagine** (§ 9.3), non a un
> ammorbidimento — ma il margine reale è ~2×, non 5-10×. Da notare anche che, essendo
> `|ΣΔ| ≤ Σ|Δ|`, **G3 scatta prima di G4**: G4 aggiunge copertura solo nel caso puramente
> compensativo per cui è stato scritto.

**G5 — Ogni scostamento grande deve essere spiegato.** Ogni SKU con `|Δ volume 24 mesi| > 5%`
**deve** avere `BestQuantile` diverso fra le due run. `BestQuantile` è il valore
post-shrinkage, cioè quello che il Modulo H applica davvero (`best_q_map`, cella 9).
Un forte scostamento **senza** un cambio di `q` non è spiegabile con il meccanismo di § 9.1:
è un bug. Questo è il gate discriminante, e non ha bisogno di essere tarato sul rumore.

> **Perché G5 non dà falsi positivi**, in forma chiusa: un flip di solo arrotondamento sposta
> il volume di 24 mesi di **un pack**, quindi `|Δ|/V > 5% ⟺ Σ₂₄(v/pack) < 20`; ma la
> probabilità che quello stesso SKU flippi è `≈ 2ε·Σ₂₄(v/pack) < 2e-5`. Le due condizioni si
> escludono a vicenda: più uno SKU è vulnerabile a un falso positivo, meno può flippare.
> Misurato: i 18 SKU su 576 con pack ≥ 5% del proprio volume hanno `Σ(v/pack)` complessivo
> 236 → **~2e-4 flip attesi fra loro**.
> *(Gli SKU esclusi dal backtest in **entrambe** le run non hanno `BestQuantile`, quindi non
> risultano "diversi": un loro `|Δ| > 5%` farebbe scattare G5. È il comportamento voluto —
> G2 impone l'identità dell'**insieme**, non del Δ per SKU — e ha probabilità ~1e-4.)*

**Diagnostica obbligatoria nel report** (riportata, non gate — serve alla firma dell'utente):
- % e numero di SKU con `BestQuantileRaw` diverso, e con `BestQuantile` diverso;
- numero di SKU con `BestAccuracyRaw == 0` (dove il `q` è arbitrario per costruzione);
- `q_global` prima e dopo;
- **KPI Motul aggregato** `Σ(BestAccuracy×TotalWeight)/Σ(TotalWeight)` prima e dopo, calcolato
  **sull'intersezione degli SKU**. È un **sanity check, non un gate**: `BestAccuracy` è
  auto-selezionato (è il valore alla `q` che quella stessa run ha scelto) e lo scostamento
  atteso è ~1e-3 punti percentuali, cioè 2-3 ordini di grandezza sotto qualunque soglia
  sensata. Se peggiorasse di oltre **0.5 pp**, però, è un segnale forte che qualcosa non va,
  e va indagato prima di proseguire;
- **top-20 per scostamento assoluto E top-20 per scostamento relativo**. Il secondo è
  obbligatorio: senza, la classe C — 399 SKU su 576 — non comparirebbe mai in nessun output
  del collaudo, e uno SKU C il cui `q` salti da 0.50 a 0.10 (−80% di forecast, rottura di
  stock su un prodotto reale) resterebbe invisibile;
- elenco completo degli SKU di **classe A o B** con `|Δ volume| > 4%`, con `q` prima/dopo.

### 9.3 Se qualcosa non torna

- **G1 fallito → è un bug del refactor.** Non è negoziabile e non ha uscite: si corregge e si
  ripete. Le opzioni qui sotto presuppongono tutte un refactor sano.
- **G2 fallito → è un bug.** Stesso trattamento.
- **G3, G4 o G5 falliti → si indaga** con la diagnostica. Se la causa è riconducibile al
  meccanismo di § 9.1 (flip di `q` su plateau) ma l'impatto eccede i gate, la decisione è
  **dell'utente**, non tecnica, e le uscite sono due, entrambe da approvare esplicitamente:
  1. **`INFERENCE_BATCH_SIZE = 1` come default di produzione**, tenendo tutto il resto: i
     numeri restano bit-identici ma si **rinuncia al miglioramento di performance
     esplicitamente richiesto**;
  2. **accettare i nuovi numeri come nuova baseline**, previa presentazione del report e
     dell'elenco degli SKU A/B interessati.

Il piano **non** prosegue ammorbidendo i gate. La scelta va registrata in § 12 e nella DoD.

---

## 10. Piano di collaudo

### 10.0 Tabella dei run *(l'ordine è obbligato)*

| Run | Codice | `INFERENCE_BATCH_SIZE` | `RUN_BACKTEST` | Ambiente | Usato in |
|---|---|---|---|---|---|
| **A** | `main` (`3f48d3c`) | 1 (di fatto: default TimesFM) | True | locale | T4.1 |
| **B** | nuovo | 1 | True | locale | T4.1 *(lato nuovo)*, **baseline di T4.2** |
| **C** | nuovo | 32 | True | locale | T4.2 |
| **B'** | nuovo | 1 | **False** | locale | baseline di T4b |
| **C'** | nuovo | 32 | **False** | locale | T4b |
| **D** | nuovo (branch di lavoro) | 1 | True | **Colab** | baseline di T5 |
| **E** | nuovo (branch di lavoro) | 32 | True | **Colab** | T5 |

Regole, da rispettare alla lettera:
- **T4.2 non si valuta finché T4.1 non è verde.** Una baseline con un bug di refactor
  renderebbe verdi confronti privi di senso.
- **Qualunque modifica al codice dopo la run B invalida B, B' e D** e impone di rifarle.
- **Precondizione della run A**: `./timesfm` deve essere già su `v2.0.2` (Fase 0) — il codice
  di `main` clona `master` se la cartella non esiste e non aggiorna mai una cartella esistente
  (`model.py:62-67`). Verificarlo con `git -C timesfm rev-parse HEAD` **prima** di lanciare A,
  e registrare anche la revision HF risolta dalla run A (dal path `snapshots/<hash>/`) per
  confermare che coincida con quella pinnata.
- Il CSV della run A non avrà `BestQuantileRaw`/`BestAccuracyRaw`/`q_global` (arrivano con la
  Fase 1.5) e va estratto a mano: `compare_forecast_outputs.py` deve tollerare insiemi di
  colonne diversi, confrontando l'intersezione.

### 10.0.1 Misura di riferimento

**Decisione (2026-08-27): la run A ufficiale si fa a valle dell'implementazione, non prima.**
Il codice di `main` resta sempre disponibile nel branch `main`, quindi la baseline non si
perde: basta tornarci per cinque minuti al momento del collaudo. Anzi, farla dopo è **più
pulito**, perché run A e run B girano sulla stessa versione TimesFM pinnata per costruzione,
senza il rischio che un aggiornamento a metà implementazione invalidi una baseline presa in
anticipo.

*(È anche il motivo pratico per cui il lavoro si fa su un branch dedicato: passare avanti e
indietro fra codice vecchio e nuovo diventa un comando, e `main` resta utilizzabile per
produrre forecast veri durante le settimane di lavoro.)*

Quello che **serve già ora** — il riferimento di tempo — è stato misurato:

Ambiente locale verificato il 2026-08-27: `.venv` con torch 2.6.0+cu124, GPU **RTX 2070
SUPER**, CUDA disponibile, tutte le dipendenze presenti, `tkinter` funzionante.

**Misura eseguita il 2026-08-27** su file Motul reale, in locale, con `./timesfm` ad
`a3beaa8` (2.0.0). Vale come riferimento di **tempo**; i valori numerici (`q_global`, KPI)
sono un riferimento indicativo, non l'artefatto formale di T4.1.

| Grandezza | Valore misurato |
|---|---|
| Device | RTX 2070 SUPER, CUDA |
| SKU nella tabella finale | **571** (shape 571×156) |
| SKU con risultato di backtest | **540** *(31 ricadono su `q = 0.5` nel Modulo H)* |
| SKU con forecast | **550** *(21 righe con colonne `f*` a `NaN` → conferma la necessità del confronto `NaN`-aware in G1)* |
| Cella 6 — Modulo F (caricamento modello) | 4,7 s |
| **Cella 7 — Modulo G (backtest, 2 origini)** | **190,9 s** |
| **Cella 8 — Modulo H (forecast futuro)** | **94,4 s** |
| Altre celle | 23,4 s |
| **Totale run** | **5 m 08,7 s** |
| `q_global` (mediana shrinkage) | **0,57** |
| KPI Motul pesato | **69,78 %** *(media semplice 51,84 %)* |

**Scomposizione della cella G** (misurata a parte, `_grid_search_cross_origin` su 540 SKU ×
2 origini con dati sintetici): **grid search in puro Python = 3,5 s**, cioè l'**1,8 %** dei
190,9 s. Il resto è inferenza. Tempo per serie coerente col benchmark di § 1.3:
187 s / 1074 chiamate = **174 ms/serie** contro i 195 ms misurati sulle serie sintetiche.

**Proiezione a batch 32** (5 ms/serie): cella G ≈ 5,4 s di inferenza + 3,5 s di grid search
≈ **9 s**; cella H ≈ **3 s**; totale run ≈ **35-40 s**.
Cioè **~5 minuti → ~40 secondi**, con ~19x sulla cella G, ~30x sulla cella H e ~8x sul tempo
totale percepito (il resto è caricamento modello e I/O, che non accelerano).

> **Correzione a § 1.3.** L'avvertenza "tolto il collo di bottiglia del modello, la grid
> search in Python diventa la quota dominante" era **una cautela eccessiva**: a 3,5 s su 191
> resta trascurabile anche dopo. Il gate resta comunque su `fl_inference_seconds`, che è la
> grandezza giusta, ma il guadagno *percepito* sarà quasi altrettanto grande.

Dopo l'implementazione si ripete la stessa run con `INFERENCE_BATCH_SIZE = 32` (run C):

**Misurato il 2026-08-27** (run B vs run C, RTX 2070 SUPER). La colonna "prima" e' la
**run B**, non la misura di riferimento di stamattina: e' quella l'unica baseline formale.

| Grandezza | Run B (batch 1) | Run C (batch 32) | Atteso | **Misurato** |
|---|---|---|---|---|
| Cella Modulo G | 194,1 s | 11,4 s | ~9 s | **17,0x** |
| Cella Modulo H | 97,1 s | 3,6 s | ~3 s | **27,0x** |
| Totale run | 5 m 13,1 s | 49,3 s | ~40 s | **6,4x** |
| `fl_inference_seconds` | 289,75 s | 9,43 s | **gate >=5x** | **30,7x — PASS** |

Lo scarto fra i 30,7x dell'inferenza e i 6,4x percepiti e' interamente nei costi fissi che il
batching non tocca: 3,5 s di grid search in Python, ~35 s fra caricamento del modello, I/O ed
Excel. La cella 6 peggiora (5,1 -> 12,9 s): compilare a batch 32 costa piu' che a batch 1, ed
e' un costo pagato una volta sola.

### T1 — Equivalenza numerica e padding *(automatico, `slow`)*

`tests/test_model_integration.py`. Modello reale caricato una volta, compilato due volte
(`per_core_batch_size` 1 e `INFERENCE_BATCH_SIZE`). Serie sintetiche deterministiche
(`np.random.RandomState(0)`): corte (6 punti = `MIN_HISTORY_POINTS`), con zeri interni,
costanti, con trend forte, con outlier.

- [x] **T1.a** `np.allclose(pf_bs1, pf_bsN, rtol=1e-4, atol=1e-3)`.
- [x] **T1.b** dopo `round_to_pack(v, pack=1, mode="nearest", decimals=3)` — l'arrotondamento
      più fine, quindi il criterio più severo — risultati identici.
- [x] **T1.c** padding: numero di serie **non multiplo** del batch (65 con batch 32),
      `len(out) == 65`.
- [x] **T1.d** il padding non contamina: con 33 serie e batch 32, verificare le righe
      **dell'ultimo chunk** (32-63 nell'indicizzazione interna, cioè la 33ª serie), non le
      prime 32 — quelle sono un batch pieno e coinciderebbero banalmente.
- [x] **T1.e** n < batch: 5 serie con batch 32 → 5 righe corrette.
- [x] **T1.f** nessuna mutazione: la lista passata all'helper ha la stessa lunghezza prima e
      dopo. *(È questo, più di T1.c, il test che protegge dal difetto di § 1.3a.)*

### T2 — Versioning, logica pura *(automatico, veloce, mock)*

- [x] `compare_versions`: `2.0.10 > 2.0.9`; `v2.0.2 == 2.0.2`; `2.1.0 > 2.0.2`;
      **segmenti di lunghezza diversa** (`v1.2 < 1.6.0`, `1.5.1 < 1.6.0`); malformato → non solleva.
- [x] Parsing di `ls-remote --tags`: **deduplica `^{}`**, tag misti, output vuoto.
- [x] `timesfm_tag("2.0.2") == "v2.0.2"`.
- [x] `latest_timesfm_tag` / `latest_lib_version`: timeout, errore git, output vuoto → `None`.
- [x] `check_library_version`: uguale → `None`; **`actual < expected` e `actual > expected`
      producono messaggi diversi e corretti**.
- [x] `check_project_updates` / `check_timesfm_update`: con `enabled=False` i mock di rete
      registrano **zero invocazioni**; con `colab=True` `local_repo_status` non viene chiamata
      ma `latest_lib_version` e `check_timesfm_update` sì.

### T2b — Versioning su repo git reali *(automatico, `slow`)*

Serve perché T2 mocka `subprocess`: non avrebbe mai potuto scoprire il problema LFS.

`ensure_timesfm_checkout` (clone reale):
- [x] clone da zero al tag → `HEAD` corretto **e** `status --porcelain` vuoto, verificato su
      `status` ripetuti e dopo un `touch` (il caso stat-cache di § Fase 0 rende un singolo
      controllo inaffidabile);
- [x] seconda chiamata su cartella già corretta → nessun comando di rete, tree pulito;
- [x] cartella su tag diverso (`v2.0.1`) → dopo la chiamata `HEAD` corretto e tree pulito;
- [x] cartella con **remote diverso** → **solleva, e la cartella esiste ancora**;
- [x] cartella non-git → errore esplicito;
- [x] cartella con tree sporco artificialmente → riclonata, tree pulito *(esercita il `rmtree`
      con handler read-only su Windows)*;
- [x] **clone fallito (URL inesistente) su cartella già presente → la cartella è ancora lì**
      (verifica dello swap da temp);
- [x] tag inesistente → solleva con `strict=True`; con `strict=False` → avviso e
      `pin_verified=False`.

`local_repo_status` e `latest_lib_version` (repo git locali temporanei, **offline**):
- [x] `git init` di un finto "remoto", clone, commit sul remoto → `behind`; commit locale →
      `ahead`; file modificato → `dirty`; `checkout HEAD~1` → detached; branch senza upstream;
      **remote diverso → nessun `fetch`**; cartella non-git → `None`.
- [x] `latest_lib_version` contro un repo locale con tag misti annotati/leggeri e a segmenti variabili.

Sequenza di allineamento del branch (§ 2.3), su cloni locali:
- [x] clone `--depth 1` di un finto remote, poi passaggio a un secondo branch con la sequenza a
      4 comandi → il branch è quello richiesto e il contenuto è aggiornato;
- [x] rieseguire la sequenza **non duplica** la refspec e non fallisce;
- [x] dopo la sequenza, `git pull` funziona (upstream impostato).

### T3 — Configurazione e fallback *(automatico, veloce, mock)*

Mock del modulo TimesFM in `sys.modules`, più `subprocess.run`,
`spec_from_file_location`/`exec_module`, `importlib.import_module` e l'`import torch` lazy.

- [x] **Asserzioni sugli argomenti passati al costruttore `ForecastConfig`, non sullo stato
      del modello**: `compile()` riscrive `max_horizon` da 24 a 128.
- [x] `per_core_batch_size` propagato; `revision` propagata a `from_pretrained`.
- [x] Non-regressione della configurazione: `normalize_inputs`, `force_flip_invariance`,
      `infer_is_positive`, `fix_quantile_crossing`, `max_context=512`, `max_horizon=horizon`.
- [x] `fl_recompile(1)` produce una config identica all'**originale** salvo `per_core_batch_size`.
- [x] **Fallback**:
      - OOM alla prima chiamata → degrado lungo `[32, 8, 1]` via `fl_recompile`; righe corrette;
        `fl_degraded = True`;
      - fallimento non-OOM in batch ma successo per singolo input → livello 3, **con degrado a
        batch 1 prima del loop**, dict errori popolato;
      - OOM nello smoke test → `fl_degraded_after_inference` resta `False`;
      - lo smoke test **non** contribuisce a `fl_inference_seconds`;
      - SKU fallito nel backtest → nessun crash, ricade su `q = 0.5`;
      - la lista di input del chiamante non viene mutata.
- [x] `run_backtest` espone `q_global` in `df.attrs` e come colonna del CSV; i tre casi in cui
      non esiste non sollevano.
- [x] `save_excel`: `run_info=None` → un foglio; con `run_info` → due fogli, **dati per primo**;
      leggibile con `pd.read_excel(path)` senza `sheet_name`.

### T4 — Regressione end-to-end su dati reali *(manuale, bloccante)*

Utility `tests/tools/compare_forecast_outputs.py`: confronta Excel + CSV di backtest + CSV
errori, calcola i gate di § 9.2 e produce la diagnostica.
- [x] **Ignora il foglio "Run info"** nel confronto (contiene data/ora e non esiste nella run A),
      ma **ne legge i campi** per il report (KPI, `q_global`, tempi).

**Setup comune**: `./timesfm` su `v2.0.2` in tutti i run, stesso file Motul reale,
`N_BACKTEST_ORIGINS = 2`, parametri di default.

- [x] **T4.1 — il refactor è neutro.** Run A vs Run B → **G1 PASS**: forecast bit-identico su
      24 colonne x 571 SKU, CSV di backtest identico su 540 SKU, **0 SKU** con `BestQuantile`
      diverso, KPI 69,7774% -> 69,7774% (+0,0000 pp). Report in `output/_report_T4.1.md`.
- [x] **T4.2 — l'effetto del batching.** Run B vs Run C → **G2, G3, G4, G5 tutti PASS**, e con
      margine totale: volume 26.623.125,5 identico al decimale (G3 +0,0000%), Sigma|Delta| = 0,0
      (G4 0,0000%), **0 SKU** con `BestQuantile` o `BestQuantileRaw` diverso, 0 SKU con
      |Delta| > 5%. Le differenze float esistono (KPI 0,6977744026 -> 0,6977735986, settima
      cifra) ma l'arrotondamento le assorbe integralmente. Report in `output/_report_T4.2.md`.
- [x] **Performance** (B vs C, `fl_inference_seconds`): 289,75 s -> 9,43 s = **30,7x su GPU**,
      contro un gate di >=5x. Cella Modulo G 194,1 -> 11,4 s; totale run 5m13,1s -> 49,3 s.
      Vedi la tabella di § 10.0.1 per la scomposizione.
- [x] Report scritto con la diagnostica di § 9.2. § 9.3 non attivato: nessun gate mancato.

### T4b — Percorso `RUN_BACKTEST = False` *(manuale)*

Percorso non esercitato da alcun test, unico consumatore di `empty_backtest_results()`.
Senza backtest tutti gli SKU usano `q = 0.5`: **nessun flip di `q`**, quindi restano solo le
differenze di arrotondamento, che § 9.1 punto 1 quantifica in **meno di un valore** sull'intero
file. I gate sono quindi molto più stretti che in T4.2.

- [x] Run B' vs Run C' → **G2, G3, G4 tutti PASS** con le soglie strette. Volume
      24.016.650,7 -> 24.016.638,7, scostamento **-0,0000%** (soglia 0,05%); Sigma|Delta| = 12,0
      su 24 milioni, rapporto **0,0000%**. Report in `output/_report_T4b.md`.
- [x] **La previsione di § 9.1 punto 1 e' confermata alla lettera**: su 13.704 celle di
      forecast (571 SKU x 24 mesi) ne differisce **esattamente una**. SKU 105880 (classe B,
      pack 12 L), mese `f2026_10`: 2016 -> 2004, cioe' **un pack**. Il valore grezzo cadeva
      sulla mezzeria del pack e la differenza float alla settima cifra lo ha fatto cadere
      dall'altra parte. E' il comportamento atteso di `round_to_pack`, non un difetto.
- [x] Fuori tolleranza → § 9.3. **Non attivato.**

### T5 — Collaudo Colab *(manuale, bloccante)*

**Prerequisito operativo.** La cella 1 clona `forecast_lib` da GitHub, non dal notebook
caricato: finché il codice nuovo non è pushato, Colab eseguirebbe la `forecast_lib` vecchia
con il notebook nuovo (`TypeError` sul kwarg `batch_size`). Quindi:
1. commit e **push su un branch di lavoro** `feat/timesfm-2.0.2`;
2. nel notebook di test `REPO_BRANCH = "feat/timesfm-2.0.2"`;
3. **riavviare il runtime** prima di rieseguire (i moduli restano in `sys.modules`);
4. eseguire le run D ed E;
5. merge su `main` **dopo** T5, `REPO_BRANCH` riportato a `"main"`.

- [ ] **T5 va eseguito su runtime GPU.** Su CPU il gate di performance non si applica (come in
      T4.2) e il collaudo non verifica il requisito dove conta.
- [ ] Run D vs Run E: **G2, G3, G4, G5** + diagnostica. *(Confronto stesso-device: fra GPU
      Colab e macchina locale le differenze float sono maggiori di quelle bs1/bs32 misurate,
      quindi un confronto cross-ambiente fallirebbe senza che nulla sia rotto.)*
- [ ] Esecuzione completa top-to-bottom senza errori; versione TimesFM caricata = `v2.0.2`,
      `fl_pin_verified = True`, revision pesi = quella pinnata (dal foglio "Run info").
- [ ] **Performance**: `fl_inference_seconds` di E ≥ 5x più veloce di D.
- [ ] Gli avvisi di aggiornamento funzionano in Colab (`check_timesfm_update` e
      `latest_lib_version` girano; `local_repo_status` no).
- [ ] Upload e download funzionano, **inclusi i due CSV di audit**; foglio "Run info" presente
      e corretto; la tabella dati è il primo foglio.
- [ ] **Dopo il merge**: uno smoke run finale in Colab da `main` con `REPO_BRANCH = "main"`
      — configurazione che nessun run precedente ha mai eseguito.

---

## 11. Definition of Done

- [x] Fasi 0-5 implementate.
- [x] `pytest` verde — **e almeno un test nuovo importa `model.py`** (oggi nessun test lo
      tocca, quindi "pytest verde" da solo non prova nulla).
- [x] `pytest -m slow` verde: T1 (a-f) e T2b (tutti i casi).
- [x] T2 e T3 verdi, incluso il gruppo sui fallback e su `save_excel`.
- [x] **T4.1 verde (G1 bit-identico)** — precondizione per valutare T4.2. *(2026-08-27)*
- [x] T4.2 eseguito, G2-G5 rispettati, report scritto con la diagnostica di § 9.2. *(2026-08-27)*
- [x] T4b eseguito, G2-G4 rispettati. *(2026-08-27)*
- [ ] T5 eseguito su Colab (runtime GPU) da branch di lavoro, tutti i criteri soddisfatti;
      smoke run finale da `main` dopo il merge.
- [x] **Code review dell'intero diff**: aderenza al piano, gestione errori, e verifica che
      `metrics.py`, `calibration.py`, `inventory.py`, `rounding.py`, `preprocessing.py`
      risultino **non toccati**.
- [ ] **Le correzioni applicate dopo la code review impongono di rieseguire `pytest` e T4.1**;
      se toccano il percorso di inferenza, anche **T4.2**; se toccano export, download o il
      percorso Colab, anche **T5** — o almeno lo smoke run finale da `main`, che è già in
      questa lista. Senza questa regola si finirebbe con correzioni mai passate da alcun
      collaudo, che è esattamente ciò che la richiesta "il piano si conclude quando il codice
      è stato rivisto, corretto **e testato**" esclude.
- [x] **Requisito performance: soddisfatto** *(30,7x su GPU, gate >=5x)* — oppure **rinunciato (§ 9.3) con approvazione
      esplicita dell'utente registrata in § 12.** Una DoD tutta spuntata con
      `INFERENCE_BATCH_SIZE = 1` e senza questa riga sarebbe un piano "riuscito" che non
      consegna ciò che è stato chiesto.
- [x] § 12 compilato e **travasato in `CLAUDE.md`**.
- [x] `forecast_lib.__version__` == `EXPECTED_FORECAST_LIB_VERSION` == tag di rilascio.
- [x] `CLAUDE.md` e `README.md` aggiornati e coerenti col codice.
- [ ] Merge su `main`, `REPO_BRANCH` riportato a `"main"`, tag di rilascio creato, commit con
      versione TimesFM pinnata e guadagno di performance misurato.
- [ ] **Questo file cancellato dal repo.**

### Stime

**Misurato il 2026-08-27** (§ 10.0.1): una run completa su file Motul reale (571 SKU) dura
**5 m 09 s**, non i "10-30 minuti" del README. Le 7 run del collaudo valgono quindi ~35 minuti
di *runtime*: il tempo del collaudo manuale è tutto nell'analisi dei diff e nella scrittura del
report, non nell'attesa. Le stime qui sotto sono tarate su questo dato.

| Fase | Stima |
|---|---|
| 0 — messa in sicurezza clone | 10 min |
| 1 — performance e robustezza | ~6 h |
| 2 — versioning | ~5 h |
| 3 — dipendenze | ~20 min |
| 4 — test automatici (incl. `compare_forecast_outputs.py`) | ~7 h |
| 5 — documentazione | ~1 h 30 |
| T4.1 + T4.2 + T4b (5 run locali da ~5 min, diff, analisi, report) | ~4 h |
| T5 (Colab: run D, E + smoke finale) | ~2 h 30 |
| Code review + correzioni + ri-collaudo | ~3 h |
| **Totale** | **~29 h → 4 giornate** |

---

## 12. Registro decisioni *(da compilare durante l'implementazione)*

| Data | Decisione | Esito / motivazione |
|---|---|---|
| 2026-08-27 | Clone pinnato vs pacchetto pip | **Clone pinnato su `v2.0.2`** (§ 3.2) |
| 2026-08-27 | Come clonare TimesFM | **`--filter=blob:none --sparse` + `sparse-checkout set src`.** Verificato su 15 cloni: unica variante con tree stabilmente pulito; `GIT_LFS_SKIP_SMUDGE` non basta (agisce sullo smudge, non sul clean) |
| 2026-08-27 | Aggiornamento del clone | **Clone in temp + swap**, con verifica del remote prima di toccare qualsiasi cosa e handler `chmod` per Windows |
| 2026-08-27 | `TIMESFM_MODEL_REVISION` | **Pinnata**. Costo zero, e il runbook la rivisita a ogni cambio di versione |
| 2026-08-27 | Verifica del pin | **Bloccante per default**, con `TIMESFM_PIN_STRICT = False` come valvola, tracciata in "Run info" e in un avviso di fine run |
| 2026-08-27 | Criteri di accettazione numerica | **Gate su identità del refactor (G1), identità strutturale (G2), impatto aggregato con e senza segno (G3/G4) e spiegabilità (G5).** Tre tentativi precedenti di gatare su conteggi di flip si sono rivelati incalibrabili: il conteggio dei flip non misura la qualità, entrambe le run scelgono un punto di un plateau |
| 2026-08-27 | KPI Motul aggregato | **Sanity check, non gate**: `BestAccuracy` è auto-selezionato e lo scostamento atteso è ~1e-3 pp, 2-3 ordini di grandezza sotto qualunque soglia sensata |
| 2026-08-27 | Baseline del confronto | **Codice nuovo a batch 1** (run B), non `main`: artefatti simmetrici e unica variabile il batch size. `main` resta coperto da T4.1 |
| 2026-08-27 | Avvisi di aggiornamento in Colab | **Attivi**: è in Colab che il pin rischia di invecchiare inosservato |
| 2026-08-27 | `q` sugli SKU con `BestAccuracyRaw == 0` | **Fuori scope**, ma registrato: oggi ricevono `q = 0.10` per effetto dell'ordine di `QUANTILE_GRID`, e **non sono confinati in classe C** (dipendono da XYZ, non da ABC). Da valutare in un ciclo dedicato |
| 2026-08-27 | Versione TimesFM effettivamente pinnata (Fase 0.0) | **`v2.0.2`, confermata.** `git ls-remote --tags` non mostra tag più recenti (ultimi: `v1.2.1`, `v1.2.6`, `v2.0.1`, `v2.0.2`): nessuna patch, minor o major nuova. Vale la riga 1 della tabella di decisione, l'analisi di § 1.2 resta valida così com'è e non serve rifare la verifica di equivalenza |
| 2026-08-27 | Revision HF effettivamente pinnata (Fase 0.0) | **`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`, invariata.** `main` su HuggingFace punta ancora a questa revision: nessun nuovo checkpoint, nessuna valutazione separata da aprire |
| 2026-08-27 | Durata misurata di una run completa | **5 m 09 s** su 571 SKU (RTX 2070 SUPER). Di cui backtest 190,9 s e forecast 94,4 s, entrambi **inferenza al ~98 %**: la grid search in Python costa 3,5 s. Proiezione a batch 32: **~40 s**. Dettagli in § 10.0.1 |
| | Valore finale di `INFERENCE_BATCH_SIZE` | *da confermare dopo T4/T5* |
| 2026-08-27 | Fase 0 (0.0 + 0.1) | **Chiusa.** Clone locale `./timesfm` ricreato pinnato su `v2.0.2` in sparse checkout (`--filter=blob:none --sparse` + `sparse-checkout set src`); entrambi i criteri di uscita verificati. `forecast_lib/` e notebook non toccati |
| 2026-08-27 | Fasi 1-3 | **Chiuse.** Modulo A (blocco parametri), `model.py` (loader esplicito, batch size, revision pinnata, attributi `fl_`, `forecast_batch_with_fallback`, smoke test bloccante), `backtest.py` (colonne `*Raw`, `q_global` in `attrs`, conteggi), `export.py` (`save_excel(run_info=)`, `build_run_info`, `save_audit_csvs`), `versioning.py` (nuovo), celle 0/1/6/7/12, igiene dipendenze. `metrics.py`, `calibration.py`, `inventory.py`, `rounding.py`, `preprocessing.py` non toccati. Test della Fase 4 non ancora scritti |
| 2026-08-27 | Nome del parametro dell'URL TimesFM | **Mantenuto `timesfm_repo_url`** (nome gia' esistente in `setup_timesfm`) invece del `repo_url` citato in § 2.2: e' keyword-only, e cambiarlo avrebbe toccato un call site senza guadagno |
| 2026-08-27 | Attributo `fl_batch_size_initial` | **Aggiunto** agli attributi di § 1.2: la scala di degrado `[N, N//4, 1]` e' derivata da `INFERENCE_BATCH_SIZE`, e dopo un degrado permanente `fl_batch_size` non e' piu' quel valore. I livelli gia' superati vengono scartati, cosi' il degrado non risale mai |
| 2026-08-27 | Come si riconosce lo smoke test in `forecast_batch_with_fallback` | **`count_time=False`**: e' l'unico chiamante che lo passa, e i due requisiti di § 1.3 (escludere il tempo, non alzare `fl_degraded_after_inference`) coincidono esattamente con quel caso. Nessun secondo parametro |
| 2026-08-27 | Composizione del foglio "Run info" e dei CSV | **In `export.py`** (`build_run_info`, `run_info_to_frame`, `save_audit_csvs`), non nella cella 12: la cella resta di sola orchestrazione e i campi diventano collaudabili con pytest. Il primo foglio conserva il nome `"Sheet1"` che pandas gli da' oggi |
| 2026-08-27 | Fase 4 | **Chiusa.** `pytest.ini` con marker `slow` e `addopts = -m "not slow"`; `test_versioning.py` (T2), `test_versioning_integration.py` (T2b), `test_model_config.py` (T3), `test_model_integration.py` (T1), casi su `save_excel` in `test_export.py`, `tests/tools/compare_forecast_outputs.py` (gate G1-G5 + diagnostica di § 9.2). **185 test veloci + 35 `slow` verdi.** T1 conferma sul modello reale l'equivalenza batch 1 / batch 32 (`rtol=1e-4`), che e' il presupposto numerico di tutto il piano |
| 2026-08-27 | Test dell'utility di confronto | **Aggiunto `tests/test_compare_forecast_outputs.py`**, non previsto dall'elenco della Fase 4: i gate G1-G5 sono lo strumento con cui si decide se il refactor e' accettabile, e un'utility che li calcola nel modo sbagliato renderebbe verde un collaudo che non lo e' |
| 2026-08-27 | Doppio di TimesFM per T3 | **Modulo `tests/_fake_timesfm.py`** eseguito davvero da `spec_from_file_location`, non un mock di `importlib`: cosi' il percorso di import di `setup_timesfm` resta coperto. Il fake **muta la lista di input** come la libreria vera, altrimenti il requisito (a) di § 1.3 non sarebbe verificabile |
| 2026-08-27 | Portata di T2b | **Un solo clone dal repo TimesFM vero** (canary dei filtri LFS, l'unica cosa che un finto remote non riproduce); tutti gli altri casi girano contro un repo locale creato con `git init`: offline, e i casi degradati (tag inesistente, remote diverso, clone fallito) diventano provocabili a comando |
| 2026-08-27 | Fase 5 | **Chiusa.** `README.md`: sezione sui parametri del modello (pin, revision, batch size, avvisi, audit), nota sul degrado e sui due flag, smoke test bloccante, foglio "Run info" e CSV, `git` come dipendenza anche per lo ZIP, nota su offline e repo promisor, runbook "Aggiornare TimesFM", `einops` -> `safetensors`, scelte progettuali, struttura, sezione test con i due gruppi, riga v1.6.0. `CLAUDE.md`: layout con `versioning.py` e `tests/tools/`, Modulo F e J aggiornati, tutti i parametri di § 4 in "Key Configuration", cinque nuove "Important Design Decisions" e travaso del registro di § 12 |
| 2026-08-27 | Code review dell'intero diff | **Eseguita** (`/code-review high` su `main...feat/timesfm-2.0.2`). `metrics.py`, `calibration.py`, `inventory.py`, `rounding.py`, `preprocessing.py` confermati **non toccati**. Sette rilievi: sei corretti, uno rimandato all'utente |
| 2026-08-27 | `fl_degraded_after_inference` (rilievo 1) | **Corretto.** Il flag si alzava a ogni degrado con `count_time=True`, incluso quello al primo tentativo della prima chiamata reale — dove nulla e' ancora stato calcolato al batch alto e il run e' quindi uniforme. Dichiarava non consegnabile un run corretto, imponendo di rifarne uno da minuti. Ora serve `fl_has_produced_output`: almeno un'inferenza reale gia' riuscita |
| 2026-08-27 | `_free_cuda_memory` nel loop per-input (rilievo 3) | **Corretto.** Era chiamata DENTRO l'`except`, contro la regola documentata dalla funzione stessa: li' l'eccezione tiene vivo il traceback e con esso i tensori che hanno esaurito la memoria, quindi `empty_cache()` non libera nulla proprio mentre si sta per chiedere la serie successiva |
| 2026-08-27 | Timeout del fetch in `local_repo_status` (rilievo 4) | **Corretto.** `TimeoutExpired` finiva nell'`except` finale e diventava `None`, che il chiamante riporta come "non e' un repository git (installazione da ZIP?)" su un clone valido. Ora il timeout degrada ad `ahead`/`behind` non calcolabili |
| 2026-08-27 | Residui dello swap (rilievo 5) | **Corretti in `.gitignore`.** `.timesfm_tmp_*` e `timesfm.old_*` non erano coperti da `timesfm/`: se la pulizia fallisce, il repository risulta sporco e il notebook avvisa a ogni avvio che il codice non corrisponde ad alcuna versione pubblicata |
| 2026-08-27 | Revision dei pesi richiesta (rilievo 6) | **Corretto.** "Run info" registrava solo la revision *risolta*, che `_resolve_model_revision` lascia a `None` quando fallisce (offline, cambio di layout della cache): in quel caso il file non conservava traccia di quali pesi fossero stati chiesti. Aggiunto il campo "Revision pesi richiesta" e il passaggio di `TIMESFM_MODEL_REVISION` dalla cella 12 |
| 2026-08-27 | Tag di pre-release (rilievo 7) | **Corretto.** `_parse_version` e' tollerante per costruzione e leggeva `v2.1.0rc1` come 2.1.0, proponendo un release candidate come aggiornamento. `_latest_tag` ora filtra con `_is_release_tag` |
| 2026-08-27 | Degrado permanente su errore non-OOM (rilievo 2) | **NON applicato: decisione dell'utente.** Un fallimento non-OOM (es. una serie malformata) fa scendere a batch 1 per tutto il resto del run, ~40x piu' lento, mentre il batch size non c'entra con la causa. Correggerlo significherebbe pero' ripristinare il batch dopo il loop per-input, cioe' contraddire la scelta esplicita di § 1.3(c) e del registro ("degrado permanente per il run"). Il rilievo 1, corretto, toglie meta' del danno: se il fallimento e' sulla prima chiamata il run non viene piu' dichiarato non consegnabile |
| 2026-08-27 | Difetti dell'utility di confronto trovati provandola su file veri | **Corretti tre.** CSV errori vuoto -> `EmptyDataError` (ed e' il caso normale: quasi nessun run ha SKU falliti); `Δ` nel report -> `UnicodeEncodeError` su console Windows cp1252, dopo aver fatto tutto il lavoro; colonne testuali (`ABC`, `XYZ`) dichiarate diverse con "0 differenze mostrate" perche' `pd.to_numeric` le riduceva a NaN. Il confronto e' ora anche indifferente al dtype (`DataFrame.equals` confronta i dtype: `int64` contro `float64` con gli stessi valori era un falso allarme garantito), restando esatto sui float |
| 2026-08-27 | Esito di T4.1 / T4.2 / T4b | **Tutti verdi.** T4.1: G1 bit-identico. T4.2: G2-G5 con scostamento aggregato esattamente 0. T4b: una sola cella su 13.704 diversa (un pack su SKU 105880), come previsto da § 9.1. Performance 30,7x contro un gate di 5x |
| | Esito di T5 | *da compilare* |
| | Eventuale attivazione di § 9.3 e scelta dell'utente | **Non attivato**: nessun gate mancato |

---

## 13. Correzioni introdotte nella v5

Rispetto alla v4, dopo due revisioni indipendenti sulla v4:

1. **§ 9 ricostruito sui gate giusti.** Il criterio 5 della v4 (cap `0.02/q` a tolleranza zero
   su A/B) sarebbe fallito quasi certamente: un flip **coarse** vale `Δq ≥ 0.05`, cioè ≥10% di
   volume, contro un cap del 4% — e sul file reale le classi A/B sono 177 SKU pari al 90,2%
   del volume. Sostituito da G4 (impatto non compensativo) e **G5 (spiegabilità)**: ogni SKU
   con `|Δ| > 5%` deve avere `BestQuantile` diverso. G5 è discriminante e non ha bisogno di
   essere tarato sul rumore.
2. **KPI Motul declassato da "criterio principale" a sanity check**, con la motivazione:
   `BestAccuracy` è auto-selezionato e lo scostamento atteso è ~1e-3 pp contro una soglia di
   0,3 pp — margine 100-1000×, quindi non poteva fallire. La garanzia reale è G1 + il bound
   `ε ≈ 5e-7`, e ora il piano lo dice esplicitamente.
3. **Numeri di § 9.1 rifatti sul file giusto**: la v4 citava `v/pack` media 836, che è il file
   dimostrativo da **30 SKU**. Sul file reale da **576 SKU** è media 109, mediana 20 — le
   stime di flip cambiano di un ordine di grandezza (da ~10% a ~1% degli SKU).
4. **Aggiunto `BestAccuracyRaw`**, simmetrico a `BestQuantileRaw`: `_apply_shrinkage`
   sovrascrive **anche** `BestAccuracy` (`backtest.py:396`) con l'accuratezza al `q`
   shrinkato, quindi con i default il valore non è un argmax e l'argomento "plateau" non era
   verificabile.
5. **Esenzione agganciata a `BestAccuracyRaw == 0`, non alla classe C**: gli SKU ad
   accuratezza nulla dipendono dall'erraticità (XYZ), non dal volume (ABC) — uno SKU A/Z è il
   candidato tipico.
6. **Aggiunta la diagnostica per scostamento relativo**: con la sola top-20 per scostamento
   assoluto, la classe C (399 SKU su 576) non sarebbe mai comparsa in alcun output del collaudo.
7. **Aggiunto G4 (`Σ|Δ|/Σ`)**: il criterio aggregato con segno si compensa, e una
   redistribuzione ampia fra SKU C sarebbe passata indenne.
8. **G2 estesa** all'insieme degli SKU con risultato di backtest: uno SKU che ne sparisce
   ricade su `q = 0.5` nel Modulo H, fino a 5× di scostamento. Rimosso "numero di SKU in
   output", invariante per costruzione (`build_final_table` fa merge `how="left"`).
9. **§ 2.3 corretto**: il ramo "branch già corretto → `git pull`" **non funziona** dopo un
   `checkout -B … FETCH_HEAD` (nessun upstream → `refusing to merge unrelated histories`), ed
   essendo non bloccante avrebbe fatto collaudare a T5 il codice vecchio in silenzio. Ora si
   esegue **sempre** la sequenza idempotente a 4 comandi, con `--set-upstream-to`, controllo
   anti-duplicazione della refspec ed esito verificato e stampato.
10. **§ 2.1.1 punto 5: clone in temp + swap.** La v4 cancellava prima di clonare: se il clone
    falliva per rete giù — lo scenario stesso per cui esiste `TIMESFM_PIN_STRICT = False` — la
    cartella non c'era più e il loader esplodeva invece di degradare.
11. **`env={}` corretto in "ambiente ereditato"**: verificato che `env={}` fa fallire il clone
    HTTPS su Windows con exit 128 (`Could not resolve host`).
12. **"Run info" e avviso di fine run svincolati da `EXPORT_AUDIT`** quando
    `pin_verified = False`: con entrambi i toggle a `False` un run non pinnato non lasciava
    alcuna traccia, riaprendo P2/P3 in silenzio.
13. **§ 9.3 distingue il fallimento di G1/G2** (bug, nessuna uscita negoziabile) da quello di
    G3/G4/G5 (decisione dell'utente).
14. **Aggiunta § 10.0, tabella dei run** con l'ordine obbligato, la regola di invalidazione di
    B/B'/D, le precondizioni della run A (clone su `v2.0.2` verificato, revision HF registrata)
    e la tolleranza sugli insiemi di colonne diversi.
15. **DoD**: le correzioni post-review impongono di rieseguire `pytest` e T4.1.
16. **Correzioni puntuali**: cap `0.02/q` = 4% a `q = 0.50` (non 2%) e sono due passi fini, non
    uno — errore rimosso con il criterio; scala di degrado espressa come `[N, N//4, 1]` derivata
    da `INFERENCE_BATCH_SIZE` e vincolata a passare da `fl_recompile` (`global_batch_size` si
    imposta solo in `compile()`); `fl_inference_seconds` esclude lo smoke test; `q_global` anche
    come colonna del CSV (`df.attrs` non sopravvive a `to_csv`); KPI aggregato fra i campi di
    "Run info"; `check_library_version` con due messaggi distinti (nella v4 quello mostrato era
    invertito); `TIMESFM_MODEL_REVISION` nel runbook "Aggiornare TimesFM"; T1.d verifica
    l'ultimo chunk, non i primi 32; T5 richiede runtime GPU; smoke run finale da `main` dopo il
    merge; confronto `NaN`-aware in G1; nota sul repo promisor nel README; obbligo di
    cronometrare una run completa prima di fissare il calendario.

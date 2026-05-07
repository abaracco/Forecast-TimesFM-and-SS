# 📦 Forecast TimesFM and SS 1.5.1

> **Previsione della domanda e pianificazione delle scorte di sicurezza** — powered by Google TimesFM-2.5-200M

[![Open In Colab](https://img.shields.io/static/v1?label=%20&message=Open%20in%20Colab&color=orange&logo=googlecolab&logoColor=F9AB00&labelColor=555)](https://colab.research.google.com/github/abaracco/Forecast-TimesFM-and-SS/blob/main/Forecast_TimesFM_and_SS.ipynb)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TimesFM](https://img.shields.io/badge/Modello-TimesFM--2.5--200M-green.svg)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![License](https://img.shields.io/badge/Licenza-MIT-lightgrey.svg)](LICENSE)

---

## 🎯 Scopo del Progetto

Questo progetto integra in un unico notebook Jupyter (eseguibile su **Google Colab** o **in locale** sul proprio PC) due ambiti della supply chain planning:

1. **Previsione della domanda** — usando il modello deep learning [TimesFM-2.5-200M](https://huggingface.co/google/timesfm-2.5-200m-pytorch) di Google, ottimizzato tramite backtest per ogni singolo SKU.
2. **Pianificazione delle scorte di sicurezza** — tramite classificazione ABC/XYZ e formula statistica standard, con livelli di servizio differenziati per classe di prodotto.

Il risultato è un file Excel completo con storico, previsioni future e metriche di inventario, pronto per essere usato nei processi S&OP e di acquisto.

---

## 🏗️ Architettura: notebook + package `forecast_lib`

A partire dalla v1.5.0 il progetto è strutturato in due livelli:

- **Notebook** (`Forecast_TimesFM_and_SS.ipynb`) — contiene la **configurazione** (Modulo A) e l'**orchestrazione** (chiamate alle funzioni nell'ordine A → J). È quello che apri e lanci.
- **Package `forecast_lib/`** — contiene tutta la **matematica della pipeline** estratta dal notebook in file `.py` brevi, leggibili e testabili. In Colab viene clonato automaticamente all'avvio del notebook (pattern `git clone`), in locale è già a fianco del notebook.

Mappa modulo → file:

| Modulo | Nome | File `forecast_lib/` | Descrizione |
|--------|------|----------------------|-------------|
| **A** | Configurazione | *(notebook, prima cella)* | Parametri globali, soglie, mappatura colonne Excel |
| **B** | Preprocessing | `preprocessing.py` | Caricamento file, rilevamento colonne temporali, conversione wide→long, filtro SKU, winsorizing |
| **C** | Serie storiche & metrica | `preprocessing.py` + `metrics.py` | Costruzione dataset di backtest; metrica di accuratezza Motul (`accuracy_single_month`, `accuracy_weighted`) |
| **D** | Calibrazione stagionale | `calibration.py` | Theil-Sen log-lineare e fattori di calibrazione per-SKU + globali |
| **E** | Arrotondamento | `rounding.py` | `round_to_pack` (`"up"` / `"down"` / `"nearest"`) |
| **F** | Modello TimesFM | `model.py` | `setup_timesfm` (loader manuale) + `forecast_all_skus_point` (batch con fallback per-SKU) |
| **G** | Backtest | `backtest.py` | `run_backtest`: rolling-origin grid search + shrinkage, senza data leakage. Disattivabile via `RUN_BACKTEST` |
| **H** | Forecast futuro | *(notebook + helpers da altri moduli)* | Pipeline scaling + calibrazione + business adjustment + arrotondamento |
| **I** | Inventario | `inventory.py` | `calculate_inventory_logic`: ABC (Pareto) + XYZ (CV) + scorta di sicurezza |
| **J** | Export | `export.py` | `build_forecast_wide`, `build_final_table`, `save_excel` |

> **Perché questa struttura?** Il notebook resta breve e leggibile (orchestrazione + grafici + risultati intermedi visibili). Le funzioni vivono in file Python normali, sono testabili con `pytest`, ricercabili dal tuo IDE, e non si appesantiscono ad ogni esecuzione del notebook. La separazione configurazione/codice rende anche più trasparente cosa è una scelta utente (Modulo A) e cosa è logica di pipeline (`forecast_lib/`).

---

## ⚙️ Come Funziona

### 1️⃣ Preprocessing (Modulo B)

Il file Excel di input ha una riga per SKU e una colonna per ogni mese di storico, in formato `YYYY_MM`. Il notebook:

- Rileva automaticamente le colonne temporali tramite pattern regex
- Converte il formato **wide → long** (una riga per SKU/mese)
- Rimuove SKU con storico insufficiente (< `MIN_HISTORY_POINTS` mesi)
- Applica **winsorizing** per tagliare i valori anomali al 5° e 95° percentile
- Elimina gli **zeri iniziali** per ogni SKU (periodo pre-lancio), mantenendo zeri interni e finali come osservazioni reali

---

### 2️⃣ Calibrazione Stagionale (Modulo D)

Per compensare i picchi/crolli ricorrenti in mesi specifici (es. agosto per ferie, dicembre per Natale), viene calcolato un **fattore moltiplicativo per-SKU** tramite regressione **Theil-Sen log-lineare**:

```
Residuo_i = log(Reale_i) − (α + β × t)
Fattore_mese = exp(mediana dei residui nei mesi target)
```

- La regressione Theil-Sen è robusta agli outlier (usa la mediana di tutte le pendenze tra coppie di punti)
- Se uno SKU ha meno di 2 osservazioni nel mese target → **fallback al fattore globale** (mediana su tutti gli SKU)
- I fattori sono **bidirezionali**: possono aumentare **o** diminuire il forecast

---

### 3️⃣ Backtest e Ottimizzazione Scaling (Modulo G)

TimesFM genera previsioni calibrate su quantili. Per trovare il quantile (e quindi il **fattore di scala**) ottimale per ogni SKU, il notebook esegue un backtest senza data leakage:

```
Per ogni origine di backtest (rolling-origin):
  Storico completo  →  Troncato (−12−shift mesi)  +  Valori reali (12 mesi)
                           ↓ TimesFM
                      Forecast × scaling_factor × fattore_stagionale → arrotondamento
                           ↓
                      Accuratezza Motul pesata per volume

Poi cross-origin:
  Per ogni q nella griglia → media accuratezza su tutte le origini
  Miglior q grossolano → raffinamento fine (step 0.01)
  Shrinkage opzionale → miscela con mediana globale
```

Il backtest include tre meccanismi di ottimizzazione:

#### a) Rolling-origin (`N_BACKTEST_ORIGINS`)

Invece di un singolo split sugli ultimi 12 mesi, il backtest valuta più finestre temporali, ognuna spostata indietro di 6 mesi. L'accuratezza per ogni scaling factor viene mediata su tutte le origini, producendo una stima più robusta e meno sensibile a periodi atipici (es. promozioni, stockout).

- `N_BACKTEST_ORIGINS = 1`: comportamento originale (singolo split)
- `N_BACKTEST_ORIGINS = 2` (default): due origini (ultimi 12 mesi + mesi da −18 a −6)
- SKU con storico insufficiente per le origini aggiuntive usano solo quelle disponibili

#### b) Griglia fine (sempre attiva)

Dopo il grid search grossolano (step 0.05, 17 punti), un secondo passaggio esplora 9 punti aggiuntivi con step 0.01 nel range `[best_q − 0.04, best_q + 0.04]`. Questo non può mai peggiorare il risultato (può solo trovare un q uguale o migliore sugli stessi dati).

#### c) Shrinkage (`SHRINKAGE_ENABLED`)

Per gli SKU con poco storico, lo scaling factor ottimale può essere instabile. Lo shrinkage miscela il q per-SKU con la mediana globale di tutti gli SKU:

```
q_finale = α × q_sku + (1 − α) × q_globale
α = min(1, mesi_storico / 36)
```

- SKU con 36+ mesi di storico → `α = 1` (nessun effetto)
- SKU con 12 mesi → `α = 0.33` (forte regolarizzazione verso la media)
- `SHRINKAGE_ENABLED = False`: disattiva completamente lo shrinkage

#### Garanzie invariate

- Tutta la calibrazione stagionale nel backtest usa **solo lo storico troncato** (no leakage)
- Per ogni SKU viene scelto il fattore che **massimizza l'accuratezza Motul pesata**
- `df_backtest_results` mantiene la stessa struttura (SKU, BestQuantile, BestAccuracy, TotalWeight)

#### d) Disattivazione totale del backtest (`RUN_BACKTEST`)

Per simulazioni rapide o per confrontare il forecast ottimizzato con una baseline non-ottimizzata, il backtest può essere completamente saltato impostando `RUN_BACKTEST = False`. In quel caso:

- L'intero Modulo G non viene eseguito (risparmio di tempo significativo, soprattutto su molti SKU)
- Tutti gli SKU usano `q = 0.5` in Modulo H, ovvero la **mediana TimesFM nativa** (ottimizzata internamente per pinball loss, non per la metrica Motul)
- Lo shrinkage non si applica (è una sotto-opzione del backtest)

Questa è una scelta forte: senza backtest il forecast non è più allineato al KPI di business Motul. Da usare consapevolmente, tipicamente solo in fase di simulazione o test metodologico.

---

### 4️⃣ Aggiustamento di Business (Modulo H)

Tra calibrazione stagionale e arrotondamento al pack, il forecast viene moltiplicato per `BUSINESS_ADJUSTMENT_FACTOR`. È una **leva manageriale di procurement**, ortogonale al modello: serve a riflettere scenari esogeni (crisi, vincoli di stock, cambi di domanda di mercato attesi) senza alterare la logica di forecasting o la sua ottimizzazione sul KPI Motul.

```
Forecast_finale = round_to_pack(
    TimesFM_q × scaling_factor × calibrazione_stagionale × BUSINESS_ADJUSTMENT_FACTOR
)
```

| Valore | Effetto |
|---|---|
| `1.0` (default) | Nessun aggiustamento — pipeline standard |
| `< 1.0` (es. `0.85`) | Abbassa il forecast (-15% nell'esempio) — utile per scenari di contrazione domanda |
| `> 1.0` (es. `1.10`) | Alza il forecast (+10%) — utile per scenari di crescita o copertura prudenziale |

> **Interazione con `ROUNDING_MODE`**: con `"up"` e fattore < 1, l'arrotondamento per eccesso può "rimangiarsi" parte della riduzione su SKU con pack grandi. Con `"nearest"` (default) l'effetto è marginale; con `"down"` la riduzione viene anzi accentuata. Comportamento atteso e coerente con la logica di procurement.

---

### 5️⃣ Metrica di Accuratezza Motul (Modulo C)

La formula di accuratezza è un requisito di business fisso e non deve essere modificata:

**Accuratezza mensile:**
```
Δ = |ACT − FCST|

ACC_i = 0   se:  ACT ≤ 0
             o:  FCST ≤ 0
             o:  FCST < ACT/2   (sotto-forecast di più della metà)
             o:  FCST > 2×ACT   (sopra-forecast di più del doppio)

ACC_i = 1 − Δ/ACT   (negli altri casi)
```

**Accuratezza pesata per volume (KPI principale):**
```
ACC = Σ( ACC_i × (ACT_i + FCST_i) ) / Σ(ACT_i + FCST_i)
```

I mesi con volumi maggiori pesano di più. Questo è il KPI reale riportato come output del backtest.

---

### 6️⃣ Classificazione ABC/XYZ e Scorta di Sicurezza (Modulo I)

#### Classificazione ABC — Pareto sui volumi

| Classe | Criteri (volume cumulativo) |
|--------|-----------------------------|
| **A** | Primi 70% del volume totale |
| **B** | Dal 70% al 90% |
| **C** | Resto (oltre il 90%) |

#### Classificazione XYZ — Volatilità della domanda

| Classe | Coefficiente di variazione (CV = σ/μ) |
|--------|----------------------------------------|
| **X** | CV ≤ 0.40 — domanda molto stabile |
| **Y** | 0.40 < CV ≤ 0.80 — domanda variabile |
| **Z** | CV > 0.80 — domanda erratica |

#### Livelli di Servizio Target per classe

| | **X** | **Y** | **Z** |
|---|---|---|---|
| **A** | 97% | 95% | 93% |
| **B** | 91% | 90% | 89% |
| **C** | 87% | 80% | **0%** (nessuna SS) |

> La classe CZ non ha scorta di sicurezza: prodotti a basso valore e domanda erratica non giustificano immobilizzi di capitale.

#### Formula Scorta di Sicurezza

```
SS = Z(SL) × σ × √( (LT + ReorderPeriod) / 30 )
```

- `Z(SL)` = z-score della distribuzione normale per il livello di servizio target
- `σ` = deviazione standard della domanda negli ultimi `SS_LOOKBACK_MONTHS` mesi
- `LT` = lead time in giorni (dalla colonna `LT` del file Excel, o default 30 gg)
- `ReorderPeriod` = periodo di riordino in giorni (fisso a 30 gg = 1 mese)

La scorta di sicurezza viene sempre arrotondata **per eccesso** al multiplo d'imballo.

---

## 📁 Formato del File di Input

Il file Excel deve avere:

| Colonna | Default | Descrizione |
|---------|---------|-------------|
| `SKU` | — | Codice prodotto (chiave univoca) |
| `Description` | — | Descrizione prodotto |
| `LT` | — | Lead time in giorni |
| `Round` | — | Multiplo d'imballo (pack size) |
| `BUn` | — | Unità di misura |
| `YYYY_MM` | — | Una colonna per ogni mese di storico (es. `2022_01`, `2022_02`, ...) |

> ℹ️ I nomi delle colonne sono configurabili nel **Modulo A**.

---

## 📊 Formato del File di Output

Il file Excel di output contiene una riga per SKU con:

1. **Metadati**: SKU, Description, Round, BUn, LT
2. **Classificazione inventario**: ABC, XYZ, SafetyStock
3. **Storico domanda**: colonne `YYYY_MM` (valori winsorizzati)
4. **Forecast futuro**: colonne `fYYYY_MM` (valori scalati + calibrati + arrotondati)

Il nome del file include automaticamente data e ora (con secondi) di estrazione: `Forecast and SS YYYY MM DD HH_MM_SS.xlsx`.

---

## 🚀 Come Iniziare

Il notebook supporta **due modalità di esecuzione**, controllate dalla variabile `COLAB` nel Modulo A:

| Modalità | `COLAB` | Input | Output |
|----------|---------|-------|--------|
| **Google Colab** (default) | `True` | Upload tramite popup del browser | Download automatico nel browser |
| **Locale** (PC) | `False` | Finestra di dialogo del sistema operativo | Salvato in `./output/` (o percorso a scelta se `ASK_SAVE_PATH = True`) |

---

### 🌐 Esecuzione su Google Colab

#### Prerequisiti

- Account Google (per Google Colab e Google Drive)
- File Excel di input nel formato descritto sopra
- Connessione internet (per il download del modello da HuggingFace, ~800 MB)

#### Passaggi

1. **Apri il notebook su Google Colab**
   - Vai su [colab.research.google.com](https://colab.research.google.com/)
   - Carica il file `Forecast_TimesFM_and_SS.ipynb`

2. **Abilita la GPU** *(raccomandato per prestazioni migliori)*
   - Menu: `Runtime` → `Change runtime type` → seleziona **T4 GPU** o superiore

3. **Esegui le celle in ordine** (dall'alto verso il basso)
   - **Modulo A**: configura i parametri se necessario (orizzonte, mesi di calibrazione, ecc.)
   - **Modulo B**: carica il file Excel quando richiesto (apparirà un pulsante di upload)
   - **Moduli C–E**: elaborazione automatica
   - **Modulo F**: installa le dipendenze e scarica il modello TimesFM (~800 MB, solo al primo avvio)
   - **Moduli G–J**: backtest, forecast, inventario e download automatico del file Excel

4. **Scarica il risultato**
   - Al termine del Modulo J, il file Excel verrà scaricato automaticamente nel browser

> ⏱️ **Tempo tipico di esecuzione**: 10–30 minuti a seconda del numero di SKU e del tipo di runtime (GPU vs CPU).

---

### 💻 Guida all'Installazione in Locale

Per eseguire il notebook sul proprio PC senza Google Colab, seguire questi passaggi:

1. **Installare Python** — Scaricare Python 3.10 o superiore da [python.org](https://www.python.org/). Su Windows, ricordare di spuntare **"Add Python to PATH"** durante l'installazione.

2. **Scaricare il repository** — Clonare il repo con `git clone <url-repo>` oppure scaricare lo ZIP da GitHub ed estrarlo.

3. **Creare un ambiente virtuale** *(raccomandato)* — Dalla cartella del progetto, aprire il terminale (o Prompt dei comandi su Windows) e lanciare:
   ```bash
   python -m venv Forecast_TimesFM_and_SS
   ```
   Poi attivarlo:
   - **Windows**: `Forecast_TimesFM_and_SS\Scripts\activate`
   - **macOS/Linux**: `source Forecast_TimesFM_and_SS/bin/activate`

4. **Installare le dipendenze** — Con l'ambiente attivato, scegliere il file adatto alla propria configurazione:

   | File | Quando usarlo |
   |------|---------------|
   | `requirements-nvidia.txt` | PC con scheda grafica **NVIDIA** e driver CUDA installati (consigliato, molto più veloce) |
   | `requirements.txt` | Tutti gli altri PC (solo CPU, più lento ma funzionante) |

   Lanciare **uno** dei due comandi:
   ```bash
   # Con GPU NVIDIA:
   pip install -r requirements-nvidia.txt

   # Senza GPU (solo CPU):
   pip install -r requirements.txt
   ```

   > **Nota CUDA**: il file `requirements-nvidia.txt` è configurato per CUDA 12.4. Se hai una versione diversa di CUDA, apri il file e cambia `cu124` con la tua versione (es. `cu121` per CUDA 12.1, `cu118` per CUDA 11.8). Per verificare la tua versione: `nvidia-smi` da terminale.

5. **Configurare il notebook** — Aprire il notebook, andare nel Modulo A e impostare `COLAB = False`.

6. **Eseguire** — Lanciare `jupyter notebook` dal terminale, aprire il file `.ipynb` e eseguire tutte le celle in ordine (`Cell → Run All`).

> **Nota**: la finestra di selezione file (tkinter) funziona con Jupyter Notebook classico. In JupyterLab o VS Code potrebbe non apparire correttamente — in tal caso, assegnare manualmente il percorso del file alla variabile `INPUT_FILE` nella cella B.1.

---

## 🔧 Configurazione (Modulo A)

Tutti i parametri si trovano nella **prima cella** del notebook (Modulo A). Sono organizzati in sezioni con commenti dettagliati. Di seguito il riepilogo.

### Toggle ON/OFF — Funzionalità matematiche

Queste variabili attivano o disattivano i passaggi matematici della pipeline. Tutte le impostazioni si applicano in modo coerente sia al backtest che al forecast futuro.

| Variabile | Default | `True` / Attivo | `False` / Disattivo |
|-----------|---------|-----------------|---------------------|
| `REMOVE_OUTLIERS` | `True` | Taglia i valori estremi (winsorizing al 5°/95° percentile) per ridurre l'impatto degli outlier | Mantiene tutti i valori originali senza filtro |
| `TRIM_LEADING_ZEROS` | `True` | Rimuove gli zeri in testa alla serie (periodo pre-lancio). Zeri interni e finali sono sempre mantenuti | Mantiene gli zeri iniziali come parte dello storico |
| `CALIBRATION_MONTHS` | `[8, 12]` | Applica un aggiustamento stagionale (Theil-Sen) ai mesi indicati (es. 8=agosto, 12=dicembre) | Impostare `[]` (lista vuota) per disattivare la calibrazione |
| `RUN_BACKTEST` | `True` | Esegue il backtest rolling-origin per trovare il quantile ottimale per SKU (massimizza l'accuratezza Motul) | Salta il backtest; tutti gli SKU usano `q = 0.5` (mediana TimesFM nativa, **non** ottimizzata sul KPI Motul) |
| `SHRINKAGE_ENABLED` | `True` | Miscela lo scaling factor di ogni SKU con la mediana globale; utile per SKU con poco storico (< 36 mesi) — *effetto solo se `RUN_BACKTEST = True`* | Usa lo scaling factor ottimale per-SKU senza correzione |
| `ROUNDING_MODE` | `"nearest"` | `"nearest"` = arrotonda al multiplo d'imballo più vicino · `"up"` = per eccesso · `"down"` = per difetto | — |
| `CALCULATE_SS` | `True` | Calcola classificazione ABC/XYZ e scorta di sicurezza | Salta il calcolo; le colonne ABC, XYZ, SafetyStock non compaiono nel file di output |

> **Nota:** la scorta di sicurezza è **sempre** arrotondata per eccesso al multiplo d'imballo, indipendentemente dal `ROUNDING_MODE` impostato per i forecast.

### Modalità di esecuzione

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `COLAB` | `True` | `True` = Google Colab (installa dipendenze automaticamente, upload/download file) · `False` = esecuzione in locale |
| `ASK_SAVE_PATH` | `False` | Solo in locale: `True` = apre finestra di dialogo per scegliere dove salvare · `False` = salva in `./output/` |

### Parametri numerici

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `HORIZON` | `24` | Mesi da prevedere nel forecast futuro (2 anni) |
| `HORIZON_BACKTEST` | `12` | Mesi della finestra di valutazione nel backtest |
| `MIN_HISTORY_POINTS` | `6` | Minimo mesi di storico richiesti per includere uno SKU |
| `N_BACKTEST_ORIGINS` | `2` | Origini di backtest (`1` = singolo split, `2`+ = rolling-origin con shift di 6 mesi) |
| `QUANTILE_GRID` | `0.10–0.90` | Griglia grossolana di ricerca dello scaling factor (step 0.05); la griglia fine (step 0.01) è automatica |
| `BUSINESS_ADJUSTMENT_FACTOR` | `1.0` | Moltiplicatore manageriale applicato al forecast finale (post backtest+calibrazione, pre-arrotondamento). `1.0` = nessun effetto, `<1.0` = abbassa, `>1.0` = alza il forecast |
| `OUTLIER_LEVEL` | `0.05` | Percentile di taglio per il winsorizing (0.05 = 5°/95°) |
| `ROUND_DECIMALS` | `3` | Decimali nel risultato finale dopo arrotondamento |
| `DEFAULT_LEAD_TIME` | `30` | Lead time di default in giorni (usato se la colonna `LT` manca nel file) |
| `REORDER_PERIOD` | `30` | Periodo di riordino in giorni (fisso a 1 mese) |
| `SS_LOOKBACK_MONTHS` | `12` | Mesi di storico usati per calcolare la deviazione standard nella scorta di sicurezza |

### Mappatura colonne Excel

Se il file di input ha nomi di colonna diversi, modificare questi valori:

| Variabile | Default | Colonna Excel |
|-----------|---------|---------------|
| `ID_COL` | `"SKU"` | Codice prodotto (chiave univoca) |
| `DESC_COL` | `"Description"` | Descrizione prodotto |
| `LT_COL_NAME` | `"LT"` | Lead time in giorni |
| `PACK_SIZE_COL` | `"Round"` | Multiplo d'imballo |
| `UOM_COL` | `"BUn"` | Unità di misura |

---

## 🧰 Dipendenze

Su **Google Colab**, le dipendenze vengono installate automaticamente dal Modulo F. In **locale**, usare i file `requirements.txt` (CPU) o `requirements-nvidia.txt` (GPU NVIDIA) forniti nel repository — vedi la [guida all'installazione](#-guida-allinstallazione-in-locale).

| Libreria | Utilizzo |
|----------|----------|
| `torch` | Backend PyTorch per TimesFM |
| `einops` | Operazioni tensoriali (richiesta da TimesFM) |
| `huggingface_hub` | Download pesi pre-addestrati |
| `pandas` | Manipolazione dati tabulari |
| `numpy` | Calcoli numerici |
| `scipy` | Distribuzione normale (z-score per scorta di sicurezza) |
| `openpyxl` | Lettura/scrittura file Excel |

> Il codice sorgente di TimesFM viene scaricato direttamente da GitHub (non tramite pip) per garantire la compatibilità con Python 3.12 di Colab.

---

## 📐 Scelte Progettuali Rilevanti

- **Nessun data leakage**: nel backtest, la calibrazione stagionale usa esclusivamente lo storico troncato, senza mai "vedere" i valori che si vuole prevedere.
- **Theil-Sen canonica unica**: la funzione `theil_sen_log_trend()` è definita una sola volta nel Modulo D e riutilizzata identicamente nel Modulo G, garantendo coerenza tra calibrazione e backtest.
- **Scaling factor ≠ quantile TimesFM**: la griglia `QUANTILE_GRID` non sfrutta l'output quantilico nativo di TimesFM (ottimizzato per pinball loss), ma viene usata come moltiplicatore della previsione mediana per massimizzare la metrica Motul.
- **Rolling-origin backtest**: più origini temporali (con shift di 6 mesi) riducono la varianza della stima dello scaling factor ottimale, rendendola più robusta a periodi atipici. La media delle accuratezze su più finestre è un'approssimazione di cross-validation per serie temporali.
- **Griglia a due passaggi**: il primo passaggio (step 0.05) identifica rapidamente la regione ottimale; il secondo (step 0.01) la affina. Il raffinamento non può mai peggiorare il risultato, solo migliorarlo.
- **Shrinkage dello scaling factor**: per SKU con storico limitato, miscela il q ottimale per-SKU con la mediana globale. Questo è un tradeoff bias-varianza classico (simile a un estimatore empirico di Bayes) che migliora la stabilità delle previsioni fuori campione.
- **Backtest disattivabile (`RUN_BACKTEST`)**: il backtest è l'unico momento in cui la metrica Motul entra esplicitamente nella scelta dei parametri. Disattivarlo significa rinunciare all'allineamento del forecast al KPI di business — è una scelta forte, da usare solo per simulazioni o confronto con baseline non-ottimizzata.
- **Aggiustamento di business separato dal modello**: `BUSINESS_ADJUSTMENT_FACTOR` agisce post-modello come moltiplicatore esplicito, mantenendo separata la logica di forecasting (modello) da quella di scenario (decisione manageriale). Questo rende le simulazioni *auditable*: l'effetto è quantificato esattamente e tracciabile, a differenza di tweak indiretti via parametri del modello.
- **Scorta di sicurezza sempre arrotondata per eccesso**: indipendentemente dal `ROUNDING_MODE` impostato per i forecast, la scorta di sicurezza usa sempre `"up"` per garantire copertura.
- **Guardia ABC**: se il volume totale nel periodo di lookback è zero, tutti gli SKU vengono classificati come classe C per evitare divisioni per zero.
- **Inferenza batch con fallback automatico**: il modello tenta prima un forecast batch (tutti gli SKU in una chiamata). Se fallisce (es. per limiti di memoria), ricade automaticamente su forecast singoli per SKU.

---

## 📂 Struttura del progetto

```
Forecast_TimesFM_and_SS/
├── Forecast_TimesFM_and_SS.ipynb   # notebook (config + orchestrazione)
├── forecast_lib/                    # matematica della pipeline
│   ├── __init__.py
│   ├── preprocessing.py             # Modulo B
│   ├── metrics.py                   # Modulo C (formula Motul)
│   ├── calibration.py               # Modulo D (Theil-Sen + fattori stagionali)
│   ├── rounding.py                  # Modulo E
│   ├── model.py                     # Modulo F (loader TimesFM + forecast batch)
│   ├── backtest.py                  # Modulo G (grid search rolling-origin)
│   ├── inventory.py                 # Modulo I (ABC/XYZ + safety stock)
│   └── export.py                    # Modulo J
├── tests/                            # test pytest sulle funzioni pure
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_rounding.py
│   ├── test_calibration.py
│   ├── test_preprocessing.py
│   ├── test_inventory.py
│   └── test_export.py
├── pytest.ini
├── requirements.txt                 # dipendenze CPU
├── requirements-nvidia.txt          # dipendenze GPU NVIDIA
├── README.md
└── CLAUDE.md
```

### Eseguire i test

I test verificano le funzioni pure del package (formula Motul, Theil-Sen, arrotondamenti, ABC/XYZ, ecc.). Sono utili per evitare regressioni durante future modifiche.

```bash
pip install pytest
pytest
```

---

## 🏷️ Storia delle versioni

| Tag | Sintesi |
|---|---|
| **v1.0** | Prima versione eseguibile su Google Colab (sviluppata senza Claude Code). |
| **v1.1** | Primi fix introdotti con l'aiuto di Claude Code. |
| **v1.2** | Ottimizzazione del codice e aggiunta di commenti (Claude Code). |
| **v1.2.1** | Aggiunto suffisso con data e ora al nome del file Excel di output. |
| **v1.3** | Metodo migliorato per aumentare l'accuracy delle previsioni. |
| **v1.4** | Algoritmo rivisto + supporto a doppia modalità di esecuzione: Google Colab **e** locale. |
| **v1.4.1** | Fix al timer della cella finale del notebook. |
| **v1.4.2** | Riorganizzazione dei parametri di configurazione nel Modulo A e aggiornamento del README. |
| **v1.4.3** | Pulizia ambiente di sviluppo: virtual environment locale rinominato in `.venv` (convenzione standard), `.gitignore` aggiornato (rimossa entry obsoleta, escluso `settings.local.json` e i file di lock di Claude Code), `settings.local.json` rimosso dal tracking git. |
| **v1.4.4** | Aggiunta `RUN_BACKTEST` per disattivare l'intero Modulo G (utile per simulazioni rapide o baseline non ottimizzata sul MAPE Motul) e `BUSINESS_ADJUSTMENT_FACTOR` come leva manageriale di procurement applicata tra calibrazione e arrotondamento (ortogonale al modello). |
| **v1.5.0** | **Refactor strutturale**: matematica della pipeline estratta dal notebook nel package `forecast_lib/` (file `.py` per modulo). Notebook ridotto da ~1960 a ~870 righe, con sole celle di configurazione e orchestrazione. Aggiunta `tests/` con suite pytest per le funzioni pure. In Colab il package viene clonato automaticamente da GitHub all'avvio (sempre ultima versione di `main`). |
| **v1.5.1** | Fix al timer di cella: la cella che registra i callback non è misurabile (il `pre_run_cell` non scatta su di essa) e viene saltata via early-return; risolto il valore spurio osservato su Colab in caso di ri-esecuzione della cella di config. Suffisso del file di output esteso ai secondi (`HH_MM_SS`) per evitare collisioni di nome su run ravvicinati. |

---

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.

---

*Progetto sviluppato con ❤️ e [Claude Code](https://claude.ai/code)*

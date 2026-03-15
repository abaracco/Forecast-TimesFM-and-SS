# 📦 Forecast & Safety Stock con TimesFM

> **Previsione della domanda e pianificazione delle scorte di sicurezza** — powered by Google TimesFM-2.5-200M

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Esegui%20su-Google%20Colab-orange.svg)](https://colab.research.google.com/)
[![TimesFM](https://img.shields.io/badge/Modello-TimesFM--2.5--200M-green.svg)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![License](https://img.shields.io/badge/Licenza-MIT-lightgrey.svg)](LICENSE)

---

## 🎯 Scopo del Progetto

Questo progetto integra in un unico notebook Jupyter (eseguibile su **Google Colab**) due ambiti della supply chain planning:

1. **Previsione della domanda** — usando il modello deep learning [TimesFM-2.5-200M](https://huggingface.co/google/timesfm-2.5-200m-pytorch) di Google, ottimizzato tramite backtest per ogni singolo SKU.
2. **Pianificazione delle scorte di sicurezza** — tramite classificazione ABC/XYZ e formula statistica standard, con livelli di servizio differenziati per classe di prodotto.

Il risultato è un file Excel completo con storico, previsioni future (fino a 25 mesi) e metriche di inventario, pronto per essere usato nei processi S&OP e di acquisto.

---

## 🏗️ Architettura: 10 Moduli (A–J)

Il notebook è organizzato in celle raggruppate per modulo funzionale:

| Modulo | Nome | Descrizione |
|--------|------|-------------|
| **A** | Configurazione | Parametri globali, soglie, mappatura colonne Excel |
| **B** | Preprocessing | Caricamento file, rilevamento colonne temporali, conversione wide→long, filtro SKU, winsorizing |
| **C** | Serie storiche | Costruzione dataset di backtest (storico troncato + valori reali); definizione della metrica di accuratezza Motul |
| **D** | Calibrazione stagionale | Calcolo fattori di aggiustamento (Theil-Sen log-lineare) per mesi critici (es. agosto, dicembre) |
| **E** | Arrotondamento | Arrotondamento al multiplo d'imballo (`"up"` / `"down"` / `"nearest"`) |
| **F** | Modello TimesFM | Caricamento manuale del modello, auto-detection GPU/CPU, smoke test, inferenza batch |
| **G** | Backtest | Grid search dello scaling factor ottimale per ogni SKU, senza data leakage |
| **H** | Forecast futuro | Generazione previsioni con scaling + calibrazione stagionale + arrotondamento |
| **I** | Inventario | Classificazione ABC (Pareto) e XYZ (CV), calcolo scorta di sicurezza |
| **J** | Export | Costruzione tabella finale e download del file Excel |

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
Storico completo  →  Troncato (−12 mesi)  +  Valori reali (ultimi 12 mesi)
                         ↓ TimesFM
                    Forecast × scaling_factor × fattore_stagionale → arrotondamento
                         ↓
                    Accuratezza Motul pesata per volume
```

- **Griglia di ricerca**: scaling factor da 0.10 a 0.90 (passo 0.05, 17 punti)
- Per ogni SKU viene scelto il fattore che **massimizza l'accuratezza Motul pesata**
- Tutta la calibrazione stagionale nel backtest usa **solo lo storico troncato** (no leakage)

---

### 4️⃣ Metrica di Accuratezza Motul (Modulo C)

La formula di accuratezza è definita dalla Casa Madre e non deve essere modificata:

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

### 5️⃣ Classificazione ABC/XYZ e Scorta di Sicurezza (Modulo I)

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

Il nome del file include automaticamente data e ora di estrazione: `Forecast and SS YYYY MM DD HH_MM.xlsx`.

---

## 🚀 Come Usare il Notebook su Google Colab

### Prerequisiti

- Account Google (per Google Colab e Google Drive)
- File Excel di input nel formato descritto sopra
- Connessione internet (per il download del modello da HuggingFace, ~800 MB)

### Passaggi

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

## 🔧 Principali Parametri di Configurazione (Modulo A)

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `HORIZON` | `25` | Mesi da prevedere |
| `HORIZON_BACKTEST` | `12` | Finestra di backtest per ottimizzazione |
| `MIN_HISTORY_POINTS` | `6` | Minimo mesi di storico per SKU |
| `REMOVE_OUTLIERS` | `True` | Abilita winsorizing |
| `OUTLIER_LEVEL` | `0.05` | Percentile di taglio (5° / 95°) |
| `CALIBRATION_MONTHS` | `[8, 12]` | Mesi con aggiustamento stagionale; `[]` per disattivare |
| `TRIM_LEADING_ZEROS` | `True` | Rimuove zeri iniziali (pre-lancio) |
| `QUANTILE_GRID` | `0.10–0.90` | Griglia di ricerca dello scaling factor |
| `ROUNDING_MODE` | `"nearest"` | Modalità arrotondamento forecast (`"up"` / `"down"` / `"nearest"`) |
| `DEFAULT_LEAD_TIME` | `30` | Lead time di default in giorni |
| `REORDER_PERIOD` | `30` | Periodo di riordino in giorni |
| `SS_LOOKBACK_MONTHS` | `12` | Mesi di storico per calcolo σ nella scorta di sicurezza |
| `CALCULATE_SS` | `True` | Abilita calcolo scorta di sicurezza |

---

## 🧰 Dipendenze

Le dipendenze vengono installate automaticamente dal **Modulo F** (`!pip install`):

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
- **Scorta di sicurezza sempre arrotondata per eccesso**: indipendentemente dal `ROUNDING_MODE` impostato per i forecast, la scorta di sicurezza usa sempre `"up"` per garantire copertura.
- **Guardia ABC**: se il volume totale nel periodo di lookback è zero, tutti gli SKU vengono classificati come classe C per evitare divisioni per zero.
- **Inferenza batch con fallback automatico**: il modello tenta prima un forecast batch (tutti gli SKU in una chiamata). Se fallisce (es. per limiti di memoria), ricade automaticamente su forecast singoli per SKU.

---

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.

---

*Progetto sviluppato con ❤️ e [Claude Code](https://claude.ai/code)*

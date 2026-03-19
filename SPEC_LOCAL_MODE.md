# Specifica: Modalità di esecuzione locale (COLAB=False)

Questo documento descrive le modifiche da apportare al notebook `Forecast_TimesFM_and_SS.ipynb` per supportare sia l'esecuzione su Google Colab che l'esecuzione in locale (Windows/Mac/Linux con Python). Il comportamento deve essere controllato da una variabile `COLAB` in Module A.

**IMPORTANTE: tutti i commenti nel codice e tutto il contenuto del README.md devono essere scritti in italiano.**

---

## Requisiti generali

Aggiungi una variabile `COLAB = True` in Module A (configurazione). Quando `COLAB = False`, il notebook deve funzionare in locale senza alcuna dipendenza da `google.colab`. Tutta la logica di forecasting (Moduli C–I) NON deve essere toccata — è già puro Python/pandas/numpy.

---

## 1. Input file (Module B — caricamento dati)

- **COLAB=True**: comportamento attuale con `google.colab.files.upload()`
- **COLAB=False**: aprire un dialog box con `tkinter.filedialog.askopenfilename()` per far selezionare il file Excel all'utente. Usare `filetypes=[("Excel files", "*.xlsx *.xls")]`. Se l'utente annulla la selezione, sollevare un errore chiaro. Nascondere la finestra principale di tkinter con `root.withdraw()`.

## 2. Output file (Module A config + Module J export)

Aggiungere una variabile `ASK_SAVE_PATH = False` in Module A.

- **COLAB=True**: comportamento attuale con `files.download(output_path)`. Ignorare `ASK_SAVE_PATH`.
- **COLAB=False + ASK_SAVE_PATH=False**: salvare nella cartella `./output/` (relativa alla directory del notebook) e stampare il percorso completo del file salvato. Creare la cartella se non esiste.
- **COLAB=False + ASK_SAVE_PATH=True**: aprire un dialog `tkinter.filedialog.asksaveasfilename()` con `defaultextension=".xlsx"` e `initialfile` basato su `OUTPUT_FILE_BASE + OUTPUT_SUFFIX`. Copiare il file nel percorso scelto dall'utente. Se annulla, fallback al salvataggio in `./output/`.

## 3. Percorsi e directory (Module A + Module F)

- **COLAB=True**: mantenere `OUTPUT_DIR = "/content/output/"` e `HF_HOME = "/content/.cache/huggingface"` come ora.
- **COLAB=False**: usare `OUTPUT_DIR = "./output/"` e lasciare `HF_HOME` al default di huggingface_hub (cioè `~/.cache/huggingface`). Non settare `HF_HOME` esplicitamente.

## 4. Installazione dipendenze (Module F — setup modello)

- **COLAB=True**: mantenere i comandi `!pip install` e `!git clone` attuali.
- **COLAB=False**: saltare `!pip install` e `!git clone`. Assumere che le dipendenze siano già installate nell'ambiente locale. Aggiungere un commento in italiano che indichi quali pacchetti devono essere installati manualmente (`pip install einops huggingface_hub torch timesfm`). Per il clone del repo TimesFM: se il path locale esiste già, non clonare di nuovo; se non esiste, clonare in una directory locale (es. `./timesfm/`) invece di `/content/timesfm`.

## 5. Aggiornamento modello (Module F)

- **COLAB=True**: non serve gestire nulla, la cache è effimera e riscarica sempre.
- **COLAB=False**: huggingface_hub fa già un ETag check automatico. Assicurarsi di NON passare `local_files_only=True` nel caricamento del modello, così che il check ETag avvenga e il modello venga aggiornato se c'è una nuova versione. Aggiungere un commento in italiano nel codice che spiega questo comportamento.

## 6. GPU/CPU

Il rilevamento GPU/CPU con `torch.cuda.is_available()` funziona già sia in Colab che in locale. Non modificare questa logica.

## 7. Import condizionali

L'import di `google.colab.files` deve avvenire SOLO quando `COLAB=True`. Wrappare tutti gli import Colab-specifici in `if COLAB:`. L'import di `tkinter` deve avvenire SOLO quando `COLAB=False`.

---

## 8. Aggiornamento README.md

Aggiornare il file `README.md` esistente (o crearlo se non esiste). **Tutto il contenuto deve essere in italiano.** Aggiungere/aggiornare le seguenti sezioni:

### Sezione "Come iniziare"

**Due modalità di esecuzione:**
- Spiegare che il notebook può girare in due modi: Google Colab (cloud, nessuna installazione richiesta) oppure in locale sul proprio PC.
- La modalità si sceglie con la variabile `COLAB` in Module A: `True` = Colab (default), `False` = locale.

**Input/Output a seconda della modalità:**
- Colab: l'input viene caricato tramite popup del browser, l'output viene scaricato automaticamente.
- Locale: l'input viene selezionato tramite finestra di dialogo del sistema operativo. L'output viene salvato nella cartella `./output/` di default, oppure in un percorso a scelta se `ASK_SAVE_PATH = True`.

### Sezione "Guida all'installazione in locale"

Guida passo-passo per far girare il notebook in locale, scritta per utenti anche non esperti:

1. **Installare Python** — Scaricare Python 3.10 o superiore da python.org. Su Windows, ricordare di spuntare "Add Python to PATH" durante l'installazione.
2. **Installare Jupyter** — Aprire il terminale (o Prompt dei comandi su Windows) e lanciare `pip install jupyter`.
3. **Scaricare il repository** — Clonare il repo con `git clone <url-repo>` oppure scaricare lo ZIP da GitHub ed estrarlo.
4. **Installare le dipendenze** — Dalla cartella del progetto, lanciare:
   ```
   pip install pandas numpy openpyxl xlsxwriter einops huggingface_hub torch timesfm
   ```
   Nota: su sistemi con GPU NVIDIA e CUDA installato, PyTorch userà automaticamente la GPU. Altrimenti funzionerà su CPU (più lento ma funzionante).
5. **Configurare il notebook** — Aprire il notebook, andare su Module A e impostare `COLAB = False`.
6. **Eseguire** — Lanciare `jupyter notebook` dal terminale, aprire il file `.ipynb` e eseguire tutte le celle in ordine (Cell → Run All).

### Sezione "Riferimento configurazione"

Aggiungere le nuove variabili `COLAB` e `ASK_SAVE_PATH` alla tabella dei parametri, con descrizione e valori di default.

---

## 9. Aggiornamento CLAUDE.md

Aggiornare il CLAUDE.md per documentare le nuove variabili `COLAB` e `ASK_SAVE_PATH` nella tabella dei parametri di Module A. Il CLAUDE.md resta in inglese perché è documentazione tecnica per Claude Code.

---

## Vincoli

- **NON** modificare la logica dei Moduli C, D, E, G, H, I (forecasting, backtest, calibrazione, safety stock)
- **NON** rinominare variabili esistenti
- Mantenere la retrocompatibilità: con `COLAB=True` il notebook deve funzionare **esattamente** come prima
- Tutti i commenti nel codice devono essere in **italiano**
- Tutto il README.md deve essere in **italiano**
- Il CLAUDE.md resta in **inglese**

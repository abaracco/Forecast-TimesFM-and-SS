"""
forecast_lib — funzioni di supporto al notebook Forecast_TimesFM_and_SS.ipynb

Ogni modulo qui dentro contiene la "matematica pura" di una fase della pipeline,
estratta dal notebook per renderlo piu' leggero e leggibile. La configurazione
(parametri, soglie, mappature colonne) resta nel Modulo A del notebook e viene
passata alle funzioni come argomenti espliciti — non esistono costanti globali
qui dentro.

Mappa modulo -> file:
    Modulo B  ->  preprocessing.py
    Modulo C  ->  metrics.py             (funzioni di accuratezza Motul)
    Modulo D  ->  calibration.py         (Theil-Sen + fattori stagionali)
    Modulo E  ->  rounding.py
    Modulo F  ->  model.py               (loader TimesFM + forecast batch)
    Modulo G  ->  backtest.py
    Modulo I  ->  inventory.py           (ABC/XYZ + scorta di sicurezza)
    Modulo J  ->  export.py              (costruzione tabella finale)
"""

__version__ = "1.0.0"

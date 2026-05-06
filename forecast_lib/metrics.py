"""
Modulo C — Funzioni di accuratezza Motul (formula aziendale, KPI di business).

Sono il cuore dell'ottimizzazione del backtest: tutto il modulo G esiste
per massimizzare `accuracy_weighted` su dati storici trattenuti. La formula
e' fissa per richiesta di Casa Madre — non modificare i 4 casi-zero.
"""


def accuracy_single_month(act, fcst):
    """
    Accuratezza mensile secondo la formula aziendale (Casa Madre):

        delta = |ACT - FCST|
        ACC_i = 0 se:
            - ACT  <= 0
            - FCST <= 0
            - delta > ACT     (forecast troppo basso: meno della meta' del reale)
            - delta > FCST    (forecast troppo alto: piu' del doppio del reale)
        altrimenti:
            1 - delta / ACT
    """
    act = float(act)
    fcst = float(fcst)

    # Condizioni che annullano l'accuratezza
    if act <= 0:
        return 0.0
    if fcst <= 0:
        return 0.0

    delta = abs(act - fcst)

    if delta > act:    # forecast < meta' del reale
        return 0.0
    if delta > fcst:   # forecast > doppio del reale
        return 0.0

    return 1 - (delta / act)


def accuracy_weighted(act_array, fcst_array):
    """
    Accuratezza pesata per volume secondo la formula aziendale:

        ACC = sum( ACC_i * (ACT_i + FCST_i) ) / sum(ACT_i + FCST_i)

    Ogni mese viene pesato per il volume (reale + previsto), cosi' i mesi
    con volumi maggiori contano di piu'.
    Restituisce 0 se i pesi totali sono nulli.
    """
    acc_list = []
    weights = []

    for act, fc in zip(act_array, fcst_array):
        acc_i = accuracy_single_month(act, fc)
        w_i = act + fc
        acc_list.append(acc_i * w_i)
        weights.append(w_i)

    total_w = sum(weights)
    if total_w <= 0:
        return 0.0

    return sum(acc_list) / total_w

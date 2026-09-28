"""
Modulo E — Arrotondamento a multiplo d'imballo (pack size).

Una sola funzione, usata sia nel forecast finale (Modulo H) che nella
scorta di sicurezza (Modulo I). I tre modi disponibili sono:
    - "up"      arrotondamento per eccesso  (richiesto per la safety stock)
    - "down"    arrotondamento per difetto
    - "nearest" arrotondamento al multiplo piu' vicino  (default per il forecast)
"""

import numpy as np
import pandas as pd


def round_to_pack(value, pack, mode="nearest", decimals=3):
    """
    Arrotonda un valore al multiplo d'imballo (pack size) piu' vicino.

    Parametri:
        value:    valore da arrotondare
        pack:     multiplo d'imballo (es. 6 -> arrotonda a 6, 12, 18, ...)
        mode:     "up" (per eccesso), "down" (per difetto), "nearest" (piu' vicino)
        decimals: numero di decimali nel risultato finale

    Se pack e' nullo o <= 0, arrotonda solo ai decimali richiesti.
    """
    if pd.isna(value):
        return value
    if pack is None or pack <= 0:
        return round(float(value), decimals) if decimals is not None else float(value)

    # Dividi per il pack, arrotonda, moltiplica per il pack
    scaled = float(value) / pack

    if mode == "up":
        scaled = np.ceil(scaled)
    elif mode == "down":
        scaled = np.floor(scaled)
    else:  # nearest
        scaled = np.round(scaled)

    result = scaled * pack
    return round(result, decimals) if decimals is not None else result

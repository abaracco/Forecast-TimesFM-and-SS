"""
Configurazione pytest: aggiunge la root del repo a sys.path
cosi' i test possono fare `from forecast_lib import ...`.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

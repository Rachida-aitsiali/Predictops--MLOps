"""Configuration de l'API, surchargeable via variables d'environnement."""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR.parent / "models" / "best_model.joblib"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", "15"))

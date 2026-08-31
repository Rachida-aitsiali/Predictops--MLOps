"""Chargement du modele et calcul des features temporelles par machine.

Le modele (Partie Maroua/Rachida) attend, en plus des valeurs capteurs brutes,
des features glissantes (lag, diff, moyennes/ecarts-types mobiles) calculees
par machine. Cette API maintient donc un petit historique en memoire par
machine_id pour reproduire exactement le feature engineering de
`train_model.py::add_temporal_features`.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone

import joblib
import pandas as pd

from app.config import HISTORY_SIZE, MODEL_PATH
from app.schemas import SensorReading

SENSOR_COLUMNS = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]


class ModelNotLoadedError(RuntimeError):
    pass


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.threshold: float = 0.5
        self.features: list[str] = []
        self.target: str | None = None
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

    def load(self) -> None:
        payload = joblib.load(MODEL_PATH)
        self.model = payload["model"]
        self.threshold = float(payload["threshold"])
        self.features = list(payload["features"])
        self.target = payload.get("target")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_type(self) -> str | None:
        if not self.is_loaded:
            return None
        try:
            return type(self.model.named_steps["classifier"]).__name__
        except Exception:
            return type(self.model).__name__

    def _rolling_features(self, machine_id: int, column: str, current_value: float) -> dict[str, float]:
        history = [entry[column] for entry in self._history[machine_id]]

        lag_1 = history[-1] if history else float("nan")
        diff_1 = current_value - lag_1 if history else float("nan")

        last_5 = history[-5:]
        last_15 = history[-15:]

        roll5_mean = statistics.fmean(last_5) if last_5 else float("nan")
        roll5_std = statistics.stdev(last_5) if len(last_5) >= 2 else float("nan")
        roll15_mean = statistics.fmean(last_15) if last_15 else float("nan")

        return {
            f"{column}_lag_1": lag_1,
            f"{column}_diff_1": diff_1,
            f"{column}_roll5_mean": roll5_mean,
            f"{column}_roll5_std": roll5_std,
            f"{column}_roll15_mean": roll15_mean,
        }

    def build_feature_row(self, reading: SensorReading) -> pd.DataFrame:
        recorded_at = reading.recorded_at or datetime.now(timezone.utc)

        row: dict[str, float | str] = {
            "machine_id": reading.machine_id,
            "temperature": reading.temperature,
            "vibration": reading.vibration,
            "humidity": reading.humidity,
            "pressure": reading.pressure,
            "energy_consumption": reading.energy_consumption,
            "recorded_hour": recorded_at.hour,
            "recorded_dayofweek": recorded_at.weekday(),
            "recorded_month": recorded_at.month,
        }

        for column in SENSOR_COLUMNS:
            row.update(self._rolling_features(reading.machine_id, column, getattr(reading, column)))

        # Historise le releve courant pour les prochaines requetes de cette machine.
        self._history[reading.machine_id].append({column: getattr(reading, column) for column in SENSOR_COLUMNS})

        frame = pd.DataFrame([row])
        return frame.reindex(columns=self.features)

    def predict(self, reading: SensorReading) -> tuple[bool, float]:
        if not self.is_loaded:
            raise ModelNotLoadedError("Le modele n'est pas charge.")

        X = self.build_feature_row(reading)
        probability = float(self.model.predict_proba(X)[0, 1])
        is_failure = probability >= self.threshold
        return is_failure, probability


model_service = ModelService()

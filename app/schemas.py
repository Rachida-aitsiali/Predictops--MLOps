"""Schemas Pydantic pour l'API de maintenance predictive."""

from datetime import datetime

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Releve capteur envoye par une machine."""

    machine_id: int = Field(..., examples=[39])
    temperature: float
    vibration: float
    humidity: float
    pressure: float
    energy_consumption: float
    recorded_at: datetime | None = Field(
        default=None,
        description="Horodatage du releve. Si absent, l'heure serveur est utilisee.",
    )


class PredictionResponse(BaseModel):
    machine_id: int
    panne_probable: bool
    probabilite_panne: float
    seuil_utilise: float
    label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None = None

"""API FastAPI - Maintenance predictive.

Partie Israe (DevOps / API Engineer) : expose le modele entraine par Maroua
(tracking MLflow de Rachida) derriere une API REST.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_service import ModelNotLoadedError, model_service
from app.schemas import HealthResponse, PredictionResponse, SensorReading


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="API de maintenance predictive",
    description="Predit la probabilite de panne d'une machine a partir de ses releves capteurs.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_type=model_service.model_type,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(reading: SensorReading) -> PredictionResponse:
    try:
        is_failure, probability = model_service.predict(reading)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(
        machine_id=reading.machine_id,
        panne_probable=is_failure,
        probabilite_panne=round(probability, 4),
        seuil_utilise=model_service.threshold,
        label="panne probable" if is_failure else "panne non probable",
    )

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client

SAMPLE_READING = {
    "machine_id": 39,
    "temperature": 75.5,
    "vibration": 2.3,
    "humidity": 40.0,
    "pressure": 101.3,
    "energy_consumption": 15.2,
}


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_prediction(client):
    response = client.post("/predict", json=SAMPLE_READING)
    assert response.status_code == 200
    body = response.json()
    assert body["machine_id"] == 39
    assert body["label"] in {"panne probable", "panne non probable"}
    assert 0.0 <= body["probabilite_panne"] <= 1.0


def test_predict_builds_history_across_calls(client):
    for _ in range(3):
        response = client.post("/predict", json=SAMPLE_READING)
        assert response.status_code == 200


def test_predict_missing_field_returns_422(client):
    incomplete = dict(SAMPLE_READING)
    del incomplete["temperature"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

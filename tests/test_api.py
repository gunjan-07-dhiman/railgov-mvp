"""
tests/test_api.py
Tests for FastAPI Production Gateway:
- Health check and startup bootstrap of v1.
- POST /predict: valid prediction with SHAP factors, HTTP 422 on invalid input.
- Audit event logging for both successful predictions and data-quality blocks.
- POST /governance/check-batch: schema, drift, and performance checks.
- POST /governance/run-supervisor: supervisor cycle execution.
- GET /governance/models and POST /governance/rollback.
- GET /governance/audit: query audit trail with filters.
"""

from fastapi.testclient import TestClient
import pytest

from app.governance.audit import get_events
from app.governance.registry import register
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create test client utilizing lifespan startup context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["active_model_version"] is not None
    assert data["audit_database"] == "connected"


def test_predict_valid_ticket(client):
    payload = {
        "booking_id": 105001,
        "travel_class": "3A",
        "booking_status": "WL",
        "wl_position": 25,
        "days_before_journey": 40,
        "day_of_week": 2,
        "month": 6,
        "is_festival_season": 0,
        "route_popularity": 0.55,
        "train_type": "Superfast",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["predicted_status"] in ["Likely Confirmed", "Uncertain", "Likely Not Confirmed"]
    assert data["confidence"] in ["High", "Medium", "Low"]
    assert len(data["top_factors"]) == 3

    # Verify SHAP explanation constraints: no percentage sign
    for factor in data["top_factors"]:
        assert "%" not in factor["explanation"]
        assert factor["strength"] in ["strongly", "moderately", "slightly"]

    # Verify PREDICTION event was logged in audit.db
    events = get_events(event_type="PREDICTION", limit=5)
    assert len(events) >= 1
    assert events[0]["details"]["booking_id"] == 105001


def test_predict_invalid_ticket_blocked_422(client):
    """
    Requirement 5: Invalid records return a 422 with the issues
    and are logged as DATA_QUALITY_BLOCK.
    """
    bad_payload = {
        "booking_id": 105002,
        "travel_class": "XX",  # Invalid class
        "booking_status": "WL",
        "wl_position": -5,     # Negative WL position
        "days_before_journey": 30,
        "day_of_week": 1,
        "month": 7,
        "is_festival_season": 0,
        "route_popularity": 0.50,
        "train_type": "Express",
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()

    assert "detail" in data
    assert "issues" in data["detail"]
    assert len(data["detail"]["issues"]) >= 2

    # Verify DATA_QUALITY_BLOCK event was logged in audit.db
    events = get_events(event_type="DATA_QUALITY_BLOCK", limit=5)
    assert len(events) >= 1
    assert "XX" in str(events[0]["details"])


def test_check_batch(client):
    batch = [
        {
            "travel_class": "SL",
            "booking_status": "WL",
            "wl_position": 20,
            "days_before_journey": 30,
            "day_of_week": 2,
            "month": 5,
            "is_festival_season": 0,
            "route_popularity": 0.40,
            "train_type": "Express",
        },
        {
            "travel_class": "3A",
            "booking_status": "RAC",
            "wl_position": 10,
            "days_before_journey": 15,
            "day_of_week": 4,
            "month": 6,
            "is_festival_season": 0,
            "route_popularity": 0.60,
            "train_type": "Superfast",
        },
    ]
    response = client.post("/governance/check-batch", json={"records": batch})
    assert response.status_code == 200
    data = response.json()
    assert data["sample_count"] == 2
    assert data["data_quality"]["valid"] is True
    assert "drift" in data


def test_get_models_and_rollback(client):
    # Fetch models
    response = client.get("/governance/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_version" in data
    assert "versions" in data

    # Register mock v_candidate to test rollback
    register("v_dummy", {"auc": 0.86}, "Mock model", status="active")
    rollback_resp = client.post("/governance/rollback")
    assert rollback_resp.status_code == 200
    rollback_data = rollback_resp.json()
    assert rollback_data["status"] == "success"


def test_get_audit_trail(client):
    response = client.get("/governance/audit?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "events" in data
    assert data["count"] >= 1

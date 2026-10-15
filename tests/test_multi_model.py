"""
tests/test_multi_model.py
Verifies multi-model governance isolation:
- Multiple model services can be registered concurrently.
- Registry catalogs and active versions are isolated per model.
- Audit logs correctly record and filter by model_name.
- Supervisor cycles execute independently for different model services.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytest

from app.base import ModelService, register_service
from app.governance.audit import get_events, init_audit_db, log_event
from app.governance.registry import get_active, list_models, list_versions, promote, register, rollback
from app.supervisor import run_supervisor_cycle


class MockDelayService(ModelService):
    """Mock secondary model service for testing multi-model isolation."""

    model_name: str = "mock_delay"
    feature_columns: List[str] = ["dist_km", "rain_mm"]
    categorical_features: List[str] = []
    numeric_features: List[str] = ["dist_km", "rain_mm"]
    target_column: str = "is_delayed"

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        missing = [c for c in self.feature_columns if c not in record]
        return len(missing) == 0, [f"Missing {c}" for c in missing]

    def validate_batch(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        missing = [c for c in self.feature_columns if c not in df.columns]
        return len(missing) == 0, [f"Missing {c}" for c in missing]

    def load_baseline_df(self) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "dist_km": rng.uniform(50, 500, 200),
            "rain_mm": rng.uniform(0, 50, 200),
            "is_delayed": rng.choice([0, 1], 200),
        })

    def train(self, df: pd.DataFrame, version: str) -> Dict[str, Any]:
        return {
            "version": version,
            "metrics": {"auc": 0.88, "accuracy": 0.82},
            "model_path": f"/tmp/{self.model_name}_{version}.joblib",
        }

    def load_model(self, version: str) -> Tuple[Any, Dict[str, Any]]:
        class DummyBundle:
            def predict_proba(self, X):
                return np.array([[0.2, 0.8]] * len(X))

            def transform(self, X):
                return X

        bundle = {"model": DummyBundle(), "preprocessor": DummyBundle(), "version": version}
        meta = {"version": version, "metrics": {"auc": 0.88}}
        return bundle, meta

    def predict(self, record: Dict[str, Any], bundle: Any) -> Dict[str, Any]:
        return {"probability": 0.8, "predicted_status": "Delayed", "confidence": "High", "top_factors": []}

    def evaluate_performance(
        self, bundle: Any, df: pd.DataFrame, baseline_metric: Optional[float] = None
    ) -> Dict[str, Any]:
        return {
            "metrics": {"auc": 0.88, "accuracy": 0.82, "f1": 0.85},
            "baseline_auc": baseline_metric or 0.88,
            "auc_drop": 0.0,
            "degradation_detected": False,
            "status": "healthy",
        }


@pytest.fixture
def delay_service():
    service = MockDelayService()
    register_service(service)
    return service


def test_multi_model_registry_isolation(tmp_path, delay_service):
    """Test that two models can register, promote, and rollback independently."""
    test_reg = tmp_path / "test_multi_reg.json"

    # Register ticket predictor version
    register("v1", {"auc": 0.84}, "Ticket training", status="active", model_name="ticket_predictor", custom_path=test_reg)

    # Register mock delay versions
    register("v1", {"auc": 0.85}, "Delay baseline", status="active", model_name="mock_delay", custom_path=test_reg)
    register("v2", {"auc": 0.90}, "Delay retrain", status="archived", model_name="mock_delay", custom_path=test_reg)

    # Verify both models are tracked separately
    models = list_models(custom_path=test_reg)
    assert "ticket_predictor" in models
    assert "mock_delay" in models

    ticket_active = get_active("ticket_predictor", custom_path=test_reg)
    delay_active = get_active("mock_delay", custom_path=test_reg)

    assert ticket_active["version"] == "v1"
    assert delay_active["version"] == "v1"

    # Promote v2 for mock_delay only
    promote("v2", model_name="mock_delay", custom_path=test_reg)
    assert get_active("mock_delay", custom_path=test_reg)["version"] == "v2"
    assert get_active("ticket_predictor", custom_path=test_reg)["version"] == "v1"

    # Rollback mock_delay only
    rollback(model_name="mock_delay", custom_path=test_reg)
    assert get_active("mock_delay", custom_path=test_reg)["version"] == "v1"
    assert get_active("ticket_predictor", custom_path=test_reg)["version"] == "v1"


def test_multi_model_audit_filtering(tmp_path):
    """Test that audit events record and filter properly per model."""
    test_db = tmp_path / "test_multi_audit.db"
    init_audit_db(test_db)

    log_event("PREDICTION", actor="tester", details={"id": 1}, model_name="ticket_predictor", db_path=test_db)
    log_event("PREDICTION", actor="tester", details={"id": 2}, model_name="mock_delay", db_path=test_db)
    log_event("DRIFT_ALERT", actor="tester", details={"id": 3}, model_name="mock_delay", db_path=test_db)

    all_events = get_events(db_path=test_db)
    ticket_events = get_events(model_name="ticket_predictor", db_path=test_db)
    delay_events = get_events(model_name="mock_delay", db_path=test_db)

    assert len(all_events) == 3
    assert len(ticket_events) == 1
    assert len(delay_events) == 2
    assert delay_events[0]["model_name"] == "mock_delay"


def test_supervisor_cycle_on_secondary_model(tmp_path, delay_service):
    """Test that supervisor executes smoothly for a non-ticket model."""
    test_reg = tmp_path / "test_sup_reg.json"
    test_db = tmp_path / "test_sup_audit.db"

    baseline = delay_service.load_baseline_df()
    batch = baseline.copy()

    report = run_supervisor_cycle(
        batch,
        service=delay_service,
        custom_registry_path=test_reg,
        custom_db_path=test_db,
        custom_baseline_df=baseline,
    )

    assert report["model_name"] == "mock_delay"
    assert report["action"] == "MONITOR"
    assert report["priority"] == "P5_HEALTHY"

    # Verify decision is logged under mock_delay
    events = get_events(model_name="mock_delay", event_type="SUPERVISOR_DECISION", db_path=test_db)
    assert len(events) >= 1
    assert events[0]["details"]["model_name"] == "mock_delay"

"""
tests/test_delay.py
Tests for Train Delay Minutes Forecasting & Platform Congestion module:
- Synthetic baseline and drifted dataset generation.
- Model training, holdout MAE / RMSE evaluation, and congestion classifier accuracy.
- Prediction output schema: error margin range (e.g. ~35 min, likely 25-45) and top 3 plain-English explanations.
- Proactive alert triggers based on delay threshold.
- Statistical covariate drift detection on weather and upstream delays.
- Autonomous supervisor incident resolution cycle for delay_predictor.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.config import (
    DATA_DIR,
    DELAY_FEATURE_COLUMNS,
    DELAY_TARGET_COLUMN,
    CONGESTION_TARGET_COLUMN,
    MODELS_DIR,
    PROACTIVE_ALERT_DELAY_THRESHOLD,
)
from app.governance.audit import get_events, init_audit_db
from app.governance.drift import detect_drift
from app.governance.registry import get_active
from app.modules.delay import delay_service, format_delay_explanation
from app.supervisor import run_supervisor_cycle
from data.generate_delay_data import generate_delay_dataset


@pytest.fixture(scope="module")
def delay_data():
    """Generate or retrieve synthetic baseline and drifted delay datasets."""
    base_df = delay_service.load_baseline_df()
    drift_df = delay_service.load_drifted_df()
    return base_df, drift_df


@pytest.fixture(scope="module")
def trained_delay_model(delay_data):
    """Train test version of delay predictor and clean up after module run."""
    base_df, _ = delay_data
    res = delay_service.train(base_df, version="test_v1")
    bundle, meta = delay_service.load_model("test_v1")
    yield bundle, meta, res

    # Clean up test artifacts
    for f in [
        MODELS_DIR / "delay_predictor_test_v1.joblib",
        MODELS_DIR / "delay_predictor_test_v1_meta.joblib",
    ]:
        if f.exists():
            f.unlink()


def test_delay_data_generation():
    """Verify synthetic delay dataset structure, types, and column presence."""
    df = generate_delay_dataset(n_samples=50, drifted=False)
    for col in DELAY_FEATURE_COLUMNS:
        assert col in df.columns
    assert DELAY_TARGET_COLUMN in df.columns
    assert CONGESTION_TARGET_COLUMN in df.columns
    assert (df[DELAY_TARGET_COLUMN] >= 0.0).all()
    assert set(df[CONGESTION_TARGET_COLUMN].unique()).issubset({"Low", "Moderate", "High"})


def test_delay_model_training_and_metrics(trained_delay_model):
    """Verify holdout MAE, R2, and platform congestion classifier accuracy."""
    _, _, res = trained_delay_model
    metrics = res["metrics"]

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "congestion_accuracy" in metrics

    # Regression MAE should be <= 12.0 mins on clean synthetic data
    assert metrics["mae"] <= 12.0
    assert metrics["r2"] >= 0.70
    assert metrics["congestion_accuracy"] >= 0.70
    assert "feature_stats" in res


def test_delay_prediction_schema_and_range(trained_delay_model):
    """Verify delay output: error margin range string, congestion tier, top factors, and alert."""
    bundle, _, _ = trained_delay_model
    sample = {
        "distance_km": 750.0,
        "scheduled_hour": 18,
        "day_of_week": 4,
        "rainfall_mm": 20.0,
        "prev_station_delay_min": 25.0,
        "route_congestion_index": 0.60,
        "train_type": "Superfast",
    }

    res = delay_service.predict(sample, bundle)

    assert "predicted_delay_minutes" in res
    assert "delay_range_display" in res
    assert "likely" in res["delay_range_display"]
    assert res["range_low"] <= res["predicted_delay_minutes"] <= res["range_high"]

    assert res["predicted_congestion"] in ["Low", "Moderate", "High"]
    assert 0.0 <= res["congestion_confidence"] <= 1.0

    assert "proactive_alert" in res
    assert len(res["top_factors"]) == 3
    for f in res["top_factors"]:
        assert "explanation" in f
        assert isinstance(f["explanation"], str)


def test_delay_proactive_alert_triggers(trained_delay_model):
    """Verify proactive alert recommendation triggers on high delays and suppresses on-time."""
    bundle, _, _ = trained_delay_model

    # High delay scenario: severe rain (60mm) and cascading upstream delay (80 min)
    high_delay_sample = {
        "distance_km": 1200.0,
        "scheduled_hour": 19,
        "day_of_week": 5,
        "rainfall_mm": 60.0,
        "prev_station_delay_min": 80.0,
        "route_congestion_index": 0.85,
        "train_type": "Express",
    }
    high_res = delay_service.predict(high_delay_sample, bundle)
    assert high_res["alert_status"] == "DELAY_ALERT"
    assert "⚠️" in high_res["proactive_alert"]
    assert high_res["predicted_delay_minutes"] >= PROACTIVE_ALERT_DELAY_THRESHOLD

    # On-time scenario: clear weather, zero upstream delay
    on_time_sample = {
        "distance_km": 100.0,
        "scheduled_hour": 11,
        "day_of_week": 1,
        "rainfall_mm": 0.0,
        "prev_station_delay_min": 0.0,
        "route_congestion_index": 0.15,
        "train_type": "Rajdhani",
    }
    low_res = delay_service.predict(on_time_sample, bundle)
    assert low_res["alert_status"] == "ON_TIME"
    assert "✅" in low_res["proactive_alert"]


def test_delay_drift_detection(delay_data):
    """Verify statistical drift flags severe distribution shift in rainfall and cascade delays."""
    base_df, drift_df = delay_data
    report = detect_drift(
        base_df,
        drift_df,
        feature_columns=delay_service.feature_columns,
        categorical_features=delay_service.categorical_features,
        numeric_features=delay_service.numeric_features,
    )

    assert report["overall_drift_detected"] is True
    assert report["overall_severity"] in ["warning", "critical"]
    metrics = report["feature_metrics"]
    assert "rainfall_mm" in metrics
    assert "prev_station_delay_min" in metrics


def test_delay_supervisor_autonomous_cycle(tmp_path, delay_data):
    """Verify autonomous supervisor executes governance, diagnoses drift, and retrains delay model."""
    test_reg = tmp_path / "delay_sup_reg.json"
    test_db = tmp_path / "delay_sup_audit.db"
    base_df, drift_df = delay_data

    report = run_supervisor_cycle(
        drift_df,
        service=delay_service,
        custom_registry_path=test_reg,
        custom_db_path=test_db,
        custom_baseline_df=base_df,
    )

    assert report["model_name"] == "delay_predictor"
    assert report["action"] in ["RETRAIN", "RETRAIN_PROMOTED", "ESCALATE_TO_HUMAN"]
    assert "drift_summary" in report

    events = get_events(model_name="delay_predictor", db_path=test_db)
    assert len(events) >= 1

"""
tests/test_fare.py
Tests for Dynamic Fare Movement & Demand Surge module:
- Synthetic baseline and drifted dataset generation.
- Model training, holdout MAE / RMSE evaluation, and demand classifier accuracy.
- Prediction output schema: rupee error margin range and top 3 plain-English explanations.
- Best booking horizon scan (0-60 days) and non-negative prospective savings.
- Residual price anomaly check for quoted fares.
- Statistical covariate drift detection (PSI > 0.20 on >= 3 features).
- Autonomous supervisor cycle execution and retraining for fare_predictor.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.config import (
    BOOKING_HORIZON_DAYS,
    DEMAND_TARGET_COLUMN,
    FARE_FEATURE_COLUMNS,
    FARE_TARGET_COLUMN,
    MODELS_DIR,
    VALID_CLASSES,
    VALID_DEMAND_LEVELS,
    VALID_TRAIN_TYPES,
)
from app.governance.audit import get_events
from app.governance.drift import detect_drift
from app.modules.fare import fare_service, format_fare_explanation
from app.supervisor import run_supervisor_cycle
from data.generate_fare_data import generate_fare_dataset


@pytest.fixture(scope="module")
def fare_data():
    """Retrieve synthetic baseline and drifted fare datasets."""
    base_df = fare_service.load_baseline_df()
    drift_df = fare_service.load_drifted_df()
    return base_df, drift_df


@pytest.fixture(scope="module")
def trained_fare_model(fare_data):
    """Train test version of fare predictor and clean up after module run."""
    base_df, _ = fare_data
    res = fare_service.train(base_df, version="test_v1")
    bundle, meta = fare_service.load_model("test_v1")
    yield bundle, meta, res

    # Clean up test artifacts
    for f in [
        MODELS_DIR / "fare_predictor_test_v1.joblib",
        MODELS_DIR / "fare_predictor_test_v1_meta.joblib",
    ]:
        if f.exists():
            f.unlink()


def test_fare_data_generation():
    """Verify synthetic fare dataset structure, types, and column presence."""
    df = generate_fare_dataset(n_samples=50, drifted=False)
    for col in FARE_FEATURE_COLUMNS:
        assert col in df.columns
    assert FARE_TARGET_COLUMN in df.columns
    assert DEMAND_TARGET_COLUMN in df.columns
    assert (df[FARE_TARGET_COLUMN] >= 50.0).all()
    assert set(df["travel_class"].unique()).issubset(set(VALID_CLASSES))
    assert set(df["train_type"].unique()).issubset(set(VALID_TRAIN_TYPES))
    assert set(df[DEMAND_TARGET_COLUMN].unique()).issubset(set(VALID_DEMAND_LEVELS))


def test_fare_model_training_and_metrics(trained_fare_model):
    """Verify holdout MAE, R2, and demand surge classifier accuracy."""
    _, _, res = trained_fare_model
    metrics = res["metrics"]

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "demand_accuracy" in metrics
    assert "auc" in metrics

    # Holdout MAE should be <= 180 INR on clean synthetic baseline data
    assert metrics["mae"] <= 180.0
    assert metrics["r2"] >= 0.85
    assert metrics["demand_accuracy"] >= 0.70


def test_fare_prediction_output_shape(trained_fare_model):
    """Verify fare output schema: range string, demand level, and top 3 explanations in rupees."""
    bundle, _, _ = trained_fare_model
    sample = {
        "travel_class": "3A",
        "train_type": "Superfast",
        "distance_km": 850.0,
        "days_before_journey": 12,
        "day_of_week": 4,
        "month": 10,
        "is_festival_season": 1,
        "route_popularity": 0.75,
        "seats_available_pct": 25.0,
    }

    res = fare_service.predict(sample, bundle)

    assert "predicted_fare_inr" in res
    assert "fare_range_display" in res
    assert "likely" in res["fare_range_display"]
    assert res["range_low"] <= res["predicted_fare_inr"] <= res["range_high"]

    assert res["predicted_demand"] in VALID_DEMAND_LEVELS
    assert 0.0 <= res["demand_confidence"] <= 1.0

    assert len(res["top_factors"]) == 3
    for f in res["top_factors"]:
        assert "explanation" in f
        assert "₹" in f["explanation"]  # TreeSHAP in rupees
        assert "%" not in f["explanation"] or "seats" in f["explanation"].lower()  # No percentage claims for prediction confidence


def test_best_booking_window(trained_fare_model):
    """Verify best booking window scans horizon and returns non-negative prospective saving."""
    bundle, _, _ = trained_fare_model
    sample = {
        "travel_class": "2A",
        "train_type": "Rajdhani",
        "distance_km": 1200.0,
        "days_before_journey": 2,  # Close to journey: fare should be high
        "day_of_week": 5,
        "month": 11,
        "is_festival_season": 1,
        "route_popularity": 0.80,
        "seats_available_pct": 15.0,
    }

    res = fare_service.best_booking_window(sample, bundle, horizon_days=BOOKING_HORIZON_DAYS)

    assert 0 <= res["best_day"] <= BOOKING_HORIZON_DAYS
    assert res["best_fare"] > 0
    assert res["today_fare"] > 0
    assert res["estimated_saving"] >= 0
    assert len(res["days_series"]) == BOOKING_HORIZON_DAYS + 1
    assert len(res["fares_series"]) == BOOKING_HORIZON_DAYS + 1


def test_price_anomaly_flag_logic():
    """Verify residual price anomaly flags quotes deviating >= 30% from predicted standard fare."""
    pred_fare = 1000.0

    # Normal quote within 10%
    normal_res = fare_service.check_price_anomaly(quoted_fare=1080.0, predicted_fare=pred_fare)
    assert normal_res["is_anomaly"] is False

    # Anomalously high quote (50% higher)
    high_res = fare_service.check_price_anomaly(quoted_fare=1500.0, predicted_fare=pred_fare)
    assert high_res["is_anomaly"] is True
    assert "higher" in high_res["message"]
    assert high_res["diff_inr"] == 500

    # Anomalously low quote (40% lower)
    low_res = fare_service.check_price_anomaly(quoted_fare=600.0, predicted_fare=pred_fare)
    assert low_res["is_anomaly"] is True
    assert "lower" in low_res["message"]
    assert low_res["diff_inr"] == 400


def test_fare_drift_detection(fare_data):
    """Verify statistical drift flags distribution shift with PSI > 0.20 on at least 3 features."""
    base_df, drift_df = fare_data
    report = detect_drift(
        base_df,
        drift_df,
        feature_columns=fare_service.feature_columns,
        categorical_features=fare_service.categorical_features,
        numeric_features=fare_service.numeric_features,
    )

    assert report["overall_drift_detected"] is True
    assert report["overall_severity"] in ["warning", "critical"]

    # Assert PSI > 0.20 on at least 3 features
    critical_features = [f for f, m in report["feature_metrics"].items() if m["psi"] >= 0.20]
    assert len(critical_features) >= 3


def test_fare_supervisor_cycle_retrains(tmp_path, fare_data):
    """Verify autonomous supervisor cycle runs on fare_predictor, detects drift, and retrains."""
    test_reg = tmp_path / "fare_sup_reg.json"
    test_db = tmp_path / "fare_sup_audit.db"
    base_df, drift_df = fare_data

    report = run_supervisor_cycle(
        drift_df,
        service=fare_service,
        custom_registry_path=test_reg,
        custom_db_path=test_db,
        custom_baseline_df=base_df,
    )

    assert report["model_name"] == "fare_predictor"
    assert report["action"] in ["RETRAIN", "RETRAIN_PROMOTED", "ESCALATE_TO_HUMAN"]
    assert "drift_summary" in report

    events = get_events(model_name="fare_predictor", db_path=test_db)
    assert len(events) >= 1

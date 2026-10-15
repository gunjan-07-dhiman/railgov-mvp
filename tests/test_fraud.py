"""
tests/test_fraud.py
Unit and governance tests for Booking Fraud Shield & Bot Detection module.
Enforces test isolation: all tests use temporary directories for models,
registry files, and audit databases so production assets are never modified.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.config import (
    FRAUD_FEATURE_COLUMNS,
    FRAUD_RISK_HIGH_THRESHOLD,
    FRAUD_RISK_LOW_THRESHOLD,
    FRAUD_TARGET_COLUMN,
    MAX_FRAUD_RECALL_DROP,
)
from app.governance.audit import get_events, init_audit_db
from app.governance.drift import detect_drift
from app.modules.fraud import format_fraud_explanation, fraud_service
from app.supervisor import run_supervisor_cycle
from data.generate_fraud_data import generate_fraud_dataset


@pytest.fixture(scope="module")
def fraud_data():
    """Generate isolated baseline and drifted datasets for test evaluation."""
    base_df = generate_fraud_dataset(n_samples=2000, fraud_rate=0.08, is_drifted=False, random_seed=42)
    drift_df = generate_fraud_dataset(n_samples=2000, fraud_rate=0.08, is_drifted=True, random_seed=101)
    return base_df, drift_df


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Provide isolated paths for models, registry, and audit db."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.modules.fraud.MODELS_DIR", models_dir)

    db_path = tmp_path / "test_audit.db"
    init_audit_db(db_path)

    reg_path = tmp_path / "test_registry.json"
    return {"models_dir": models_dir, "db_path": db_path, "reg_path": reg_path}


def test_fraud_data_generation(fraud_data):
    """Verify synthetic dataset shapes, column types, fraud rate, and privacy guarantee."""
    base_df, drift_df = fraud_data

    assert len(base_df) == 2000
    assert len(drift_df) == 2000

    for col in FRAUD_FEATURE_COLUMNS + [FRAUD_TARGET_COLUMN]:
        assert col in base_df.columns
        assert col in drift_df.columns

    # Verify fraud rate ~8%
    assert 0.07 <= base_df[FRAUD_TARGET_COLUMN].mean() <= 0.09
    assert 0.07 <= drift_df[FRAUD_TARGET_COLUMN].mean() <= 0.09

    # Verify privacy guarantee: no raw IP or device ID columns
    for col in base_df.columns:
        assert "ip_address" not in col.lower()
        assert "mac_address" not in col.lower()
        assert "device_id" not in col.lower()


def test_fraud_model_training_and_metrics(fraud_data, isolated_env):
    """Verify training, holdout recall/precision within expected target bands, and no NaN."""
    base_df, _ = fraud_data
    res = fraud_service.train(base_df, version="test_v1")

    assert res["version"] == "test_v1"
    metrics = res["metrics"]
    assert "recall" in metrics
    assert "precision" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics
    assert "auc" in metrics

    # Verify baseline holdout targets: ~85-95% recall and ~70-90% precision with overlapping hard cases
    assert 0.80 <= metrics["recall"] <= 0.98
    assert 0.65 <= metrics["precision"] <= 0.95
    assert metrics["fpr"] < 0.08
    assert metrics["fnr"] < 0.20


def test_fraud_predict_output_schema_and_reasons(fraud_data, isolated_env):
    """Verify prediction risk score, tiers, TreeSHAP drivers, and STRICT no percentage sign rule."""
    base_df, _ = fraud_data
    fraud_service.train(base_df, version="test_v1")
    bundle, meta = fraud_service.load_model("test_v1")

    # 1. Normal human booking session
    human_session = {
        "form_fill_seconds": 38.0,
        "bookings_last_hour": 1,
        "accounts_per_device": 1,
        "accounts_per_ip": 1,
        "failed_payments": 0,
        "account_age_days": 210,
        "is_tatkal_window": 0,
        "avg_gap_between_clicks_ms": 1450.0,
    }
    res_human = fraud_service.predict(human_session, bundle, custom_db_path=isolated_env["db_path"])

    assert 0 <= res_human["risk_score"] <= 100
    assert res_human["risk_level"] in ["Low", "Medium", "High"]
    assert len(res_human["reasons"]) == 3

    # STRICT RULE: Explanations must never contain percentage signs
    for r in res_human["reasons"]:
        assert "%" not in r, f"Percentage sign found in explanation: {r}"
        assert isinstance(r, str)

    # 2. Automated rapid bot session
    bot_session = {
        "form_fill_seconds": 1.2,
        "bookings_last_hour": 10,
        "accounts_per_device": 6,
        "accounts_per_ip": 5,
        "failed_payments": 2,
        "account_age_days": 2,
        "is_tatkal_window": 1,
        "avg_gap_between_clicks_ms": 110.0,
    }
    res_bot = fraud_service.predict(bot_session, bundle, custom_db_path=isolated_env["db_path"])

    assert res_bot["risk_score"] >= FRAUD_RISK_HIGH_THRESHOLD
    assert res_bot["risk_level"] == "High"
    assert res_bot["status_light"] == "red"
    for r in res_bot["reasons"]:
        assert "%" not in r


def test_fraud_audit_events_isolation(fraud_data, isolated_env):
    """Verify audit logs: PREDICTION_MADE on every session, HIGH_RISK_ALERT on High risk sessions."""
    base_df, _ = fraud_data
    fraud_service.train(base_df, version="test_v1")
    bundle, _ = fraud_service.load_model("test_v1")

    # Score normal session
    human_session = {
        "form_fill_seconds": 45.0,
        "bookings_last_hour": 0,
        "accounts_per_device": 1,
        "accounts_per_ip": 1,
        "failed_payments": 0,
        "account_age_days": 350,
        "is_tatkal_window": 0,
        "avg_gap_between_clicks_ms": 1800.0,
    }
    fraud_service.predict(human_session, bundle, custom_db_path=isolated_env["db_path"])

    events_pred = get_events(
        event_type="PREDICTION_MADE", model_name="fraud_detector", db_path=isolated_env["db_path"]
    )
    assert len(events_pred) == 1
    assert events_pred[0]["model_name"] == "fraud_detector"

    # Score bot session to trigger high risk alert
    bot_session = {
        "form_fill_seconds": 0.9,
        "bookings_last_hour": 12,
        "accounts_per_device": 7,
        "accounts_per_ip": 5,
        "failed_payments": 3,
        "account_age_days": 1,
        "is_tatkal_window": 1,
        "avg_gap_between_clicks_ms": 95.0,
    }
    fraud_service.predict(bot_session, bundle, custom_db_path=isolated_env["db_path"])

    events_alert = get_events(
        event_type="HIGH_RISK_ALERT", model_name="fraud_detector", db_path=isolated_env["db_path"]
    )
    assert len(events_alert) == 1
    assert events_alert[0]["model_name"] == "fraud_detector"
    assert events_alert[0]["details"]["risk_score"] >= FRAUD_RISK_HIGH_THRESHOLD


def test_fraud_covariate_drift_and_degradation(fraud_data, isolated_env):
    """Verify statistical drift and holdout recall degradation on drifted dataset."""
    base_df, drift_df = fraud_data
    fraud_service.train(base_df, version="test_v1")
    bundle, meta = fraud_service.load_model("test_v1")

    # 1. Covariate drift detection
    drift_report = detect_drift(
        base_df,
        drift_df,
        feature_columns=fraud_service.feature_columns,
        categorical_features=fraud_service.categorical_features,
        numeric_features=fraud_service.numeric_features,
    )
    assert drift_report["overall_drift_detected"] is True
    assert drift_report["critical_count"] >= 1

    # 2. Performance degradation evaluation
    baseline_rec = meta["metrics"]["recall"]
    eval_res = fraud_service.evaluate_performance(bundle, drift_df, baseline_metric=baseline_rec)

    assert eval_res["recall_drop"] > MAX_FRAUD_RECALL_DROP
    assert eval_res["degradation_detected"] is True
    assert eval_res["status"] == "degraded"


def test_fraud_supervisor_autonomous_cycle(fraud_data, isolated_env):
    """Verify supervisor detects fraud degradation, logs PERFORMANCE_ALERT, and triggers retraining."""
    base_df, drift_df = fraud_data

    report = run_supervisor_cycle(
        drift_df,
        service=fraud_service,
        custom_registry_path=isolated_env["reg_path"],
        custom_db_path=isolated_env["db_path"],
        custom_baseline_df=base_df,
    )

    assert report["model_name"] == "fraud_detector"
    assert report["action"] in ["RETRAIN", "RETRAIN_PROMOTED", "ESCALATE_TO_HUMAN"]
    assert "drift_summary" in report
    assert "performance_summary" in report

    events = get_events(model_name="fraud_detector", db_path=isolated_env["db_path"])
    assert len(events) >= 2  # supervisor decisions and alerts logged

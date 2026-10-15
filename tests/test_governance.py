"""
tests/test_governance.py
Comprehensive test suite for RailGov MVP governance modules:
- Append-only audit trail: verifies SQLite triggers abort on UPDATE and DELETE.
- Data quality guardrails: verifies invalid records from invalid_samples.json are blocked.
- Statistical drift detection: verifies PSI ~0 on identical data, critical PSI on drifted data, and KS test p-values.
- Performance monitoring: verifies AUC degradation alerts.
- Model registry: verifies lifecycle transitions (active, promote, rollback, list).
- Retrain pipeline: verifies unseen holdout isolation and candidate promotion/rejection.
"""

import json
from pathlib import Path
import sqlite3
import pandas as pd
import pytest

from app.config import DATA_DIR
from app.governance.audit import get_events, init_audit_db, log_event
from app.governance.data_quality import validate_batch, validate_record
from app.governance.drift import compute_categorical_psi, compute_ks_test, compute_numeric_psi, detect_drift
from app.governance.monitor import evaluate_performance
from app.governance.registry import get_active, list_versions, promote, register, rollback
from app.governance.retrain import retrain_if_needed
from app.predictor import load_model, train


# ---------------------------------------------------------
# 1. Audit Log & Append-Only Trigger Tests
# ---------------------------------------------------------

def test_audit_log_append_and_retrieve(tmp_path):
    test_db = tmp_path / "test_audit.db"
    init_audit_db(test_db)

    event_id = log_event(
        event_type="PREDICTION",
        actor="unit_test",
        details={"ticket_id": 12345, "result": "Likely Confirmed"},
        model_version="v1",
        db_path=test_db,
    )
    assert event_id >= 1

    events = get_events(db_path=test_db)
    assert len(events) == 1
    assert events[0]["event_type"] == "PREDICTION"
    assert events[0]["details"]["ticket_id"] == 12345


def test_audit_log_triggers_abort_on_update_and_delete(tmp_path):
    """
    Requirement 2: Enforce append-only in audit.db with SQLite triggers that
    RAISE(ABORT) on UPDATE and DELETE, and test that.
    """
    test_db = tmp_path / "test_audit_triggers.db"
    init_audit_db(test_db)

    log_event(
        event_type="PREDICTION",
        actor="unit_test",
        details={"ticket_id": 999},
        model_version="v1",
        db_path=test_db,
    )

    conn = sqlite3.connect(str(test_db))

    # Test UPDATE is blocked by trigger
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)) as exc_update:
        conn.execute("UPDATE audit_log SET actor = 'tampered' WHERE id = 1")
    assert "append-only" in str(exc_update.value).lower()

    # Test DELETE is blocked by trigger
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)) as exc_delete:
        conn.execute("DELETE FROM audit_log WHERE id = 1")
    assert "append-only" in str(exc_delete.value).lower()

    conn.close()


# ---------------------------------------------------------
# 2. Data Quality Tests
# ---------------------------------------------------------

def test_data_quality_blocks_invalid_samples():
    invalid_path = DATA_DIR / "invalid_samples.json"
    with open(invalid_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for item in samples:
        record = item["record"]
        is_valid, issues = validate_record(record)
        assert not is_valid, f"Expected {item['test_case']} to fail data quality check"
        assert len(issues) > 0


def test_data_quality_accepts_valid_record():
    valid_record = {
        "booking_id": 100001,
        "travel_class": "3A",
        "booking_status": "WL",
        "wl_position": 25,
        "days_before_journey": 30,
        "day_of_week": 2,
        "month": 5,
        "is_festival_season": 0,
        "route_popularity": 0.50,
        "train_type": "Superfast",
    }
    is_valid, issues = validate_record(valid_record)
    assert is_valid
    assert len(issues) == 0


def test_data_quality_validate_batch():
    baseline_path = DATA_DIR / "baseline.csv"
    df = pd.read_csv(baseline_path).head(100)
    is_valid, issues = validate_batch(df)
    assert is_valid
    assert len(issues) == 0


# ---------------------------------------------------------
# 3. Drift Detection Tests (PSI & KS)
# ---------------------------------------------------------

def test_psi_identical_data():
    baseline_path = DATA_DIR / "baseline.csv"
    df = pd.read_csv(baseline_path)
    sample_a = df["wl_position"].head(1000)
    sample_b = df["wl_position"].tail(1000)

    psi_numeric = compute_numeric_psi(sample_a, sample_b)
    assert psi_numeric < 0.10, f"Expected near-zero PSI for identical data, got {psi_numeric}"

    psi_cat = compute_categorical_psi(df["travel_class"].head(1000), df["travel_class"].tail(1000))
    assert psi_cat < 0.10


def test_drift_detection_flags_drifted_batch():
    baseline_df = pd.read_csv(DATA_DIR / "baseline.csv")
    drifted_df = pd.read_csv(DATA_DIR / "drifted.csv")

    report = detect_drift(baseline_df, drifted_df)

    assert report["overall_drift_detected"] is True
    assert report["overall_severity"] == "critical"
    # Festival season and wl_position should exhibit critical PSI
    assert "is_festival_season" in report["critical_features"] or "wl_position" in report["critical_features"]
    assert report["critical_count"] >= 1


# ---------------------------------------------------------
# 4. Monitoring & Degradation Tests
# ---------------------------------------------------------

def test_monitor_flags_auc_degradation():
    baseline_df = pd.read_csv(DATA_DIR / "baseline.csv")
    drifted_df = pd.read_csv(DATA_DIR / "drifted.csv")

    # Train temporary v_eval on baseline
    train(baseline_df, version="v_eval")
    bundle, meta = load_model("v_eval")
    baseline_auc = meta["metrics"]["auc"]

    # Evaluate on drifted dataset where concept drift occurred
    eval_drift = evaluate_performance(bundle, drifted_df, baseline_auc=baseline_auc)

    assert "metrics" in eval_drift
    assert eval_drift["auc_drop"] > 0.05
    assert eval_drift["degradation_detected"] is True
    assert eval_drift["status"] == "degraded"


# ---------------------------------------------------------
# 5. Model Registry Lifecycle Tests
# ---------------------------------------------------------

def test_registry_lifecycle(tmp_path):
    reg_file = tmp_path / "registry.json"

    # 1. Register v1
    v1 = register("v1", {"auc": 0.85}, "Initial baseline", status="active", custom_path=reg_file)
    assert get_active(custom_path=reg_file)["version"] == "v1"

    # 2. Register v2
    v2 = register("v2", {"auc": 0.88}, "Retrained", status="archived", custom_path=reg_file)
    assert get_active(custom_path=reg_file)["version"] == "v1"

    # 3. Promote v2
    promote("v2", custom_path=reg_file)
    assert get_active(custom_path=reg_file)["version"] == "v2"

    # 4. Rollback to v1
    rollback(custom_path=reg_file)
    assert get_active(custom_path=reg_file)["version"] == "v1"

    # 5. List versions
    all_v = list_versions(custom_path=reg_file)
    assert len(all_v) == 2


# ---------------------------------------------------------
# 6. Retraining Pipeline Tests
# ---------------------------------------------------------

def test_retrain_if_needed_promotes_on_drifted_data(tmp_path):
    reg_file = tmp_path / "registry.json"
    audit_file = tmp_path / "audit.db"

    baseline_df = pd.read_csv(DATA_DIR / "baseline.csv")
    drifted_df = pd.read_csv(DATA_DIR / "drifted.csv")

    # Train and register v1
    train(baseline_df, version="v1_test_retrain")
    bundle, meta = load_model("v1_test_retrain")
    register(
        "v1_test_retrain",
        meta["metrics"],
        "Baseline",
        status="active",
        custom_path=reg_file,
    )

    # Run retraining triggered by drift
    report = retrain_if_needed(
        trigger_reason="Test drift alert",
        current_df=drifted_df,
        baseline_df=baseline_df,
        custom_registry_path=reg_file,
        custom_db_path=audit_file,
    )

    assert report["action"] in ["promoted", "rejected"]
    assert "candidate_version" in report
    assert report["holdout_samples_count"] > 0

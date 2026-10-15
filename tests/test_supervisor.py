"""
tests/test_supervisor.py
Tests for Rule-Based Supervisor Agent:
- Priority ranking (Critical Drift + Degradation > Normal).
- Root Cause Analysis (identifies top drifted features and links to metrics).
- Autonomous retraining and candidate promotion on drifted data.
- Escalation to human operators when retraining criteria fail.
- Guardrail halting on corrupt data batches.
"""

from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from app.config import DATA_DIR
from app.governance.audit import get_events, init_audit_db
from app.governance.registry import get_active, register
from app.predictor import load_model, train
from app.supervisor import run_supervisor_cycle
from data.generate_data import generate_bookings


@pytest.fixture
def supervisor_env(tmp_path):
    """Set up isolated model registry and audit database for supervisor testing."""
    reg_path = tmp_path / "registry.json"
    audit_path = tmp_path / "audit.db"
    init_audit_db(audit_path)

    baseline_df = pd.read_csv(DATA_DIR / "baseline.csv")
    train(baseline_df, version="v1_sup_test")
    bundle, meta = load_model("v1_sup_test")
    register(
        "v1_sup_test",
        meta["metrics"],
        "Baseline training",
        status="active",
        custom_path=reg_path,
    )

    return {
        "reg_path": reg_path,
        "audit_path": audit_path,
        "baseline_df": baseline_df,
    }


def test_supervisor_cycle_healthy_batch(supervisor_env):
    # Batch generated with normal operations (different seed from training)
    normal_batch = generate_bookings(n=500, drifted=False, seed=777)

    report = run_supervisor_cycle(
        current_batch_df=normal_batch,
        custom_registry_path=supervisor_env["reg_path"],
        custom_db_path=supervisor_env["audit_path"],
        custom_baseline_df=supervisor_env["baseline_df"],
    )

    assert report["action"] == "MONITOR"
    assert report["priority"] in ["P4_DRIFT_WARNING", "P5_HEALTHY"]
    assert "healthy" in report["root_cause"].lower() or "within warning" in report["root_cause"].lower()

    # Verify audit event logged
    events = get_events(event_type="SUPERVISOR_DECISION", db_path=supervisor_env["audit_path"])
    assert len(events) >= 1
    assert events[0]["details"]["action"] == "MONITOR"


def test_supervisor_cycle_drifted_batch_triggers_retrain(supervisor_env):
    drifted_df = pd.read_csv(DATA_DIR / "drifted.csv")

    report = run_supervisor_cycle(
        current_batch_df=drifted_df,
        custom_registry_path=supervisor_env["reg_path"],
        custom_db_path=supervisor_env["audit_path"],
        custom_baseline_df=supervisor_env["baseline_df"],
    )

    # Must prioritize drift & degradation, and execute retraining
    assert report["priority"] in [
        "P1_CRITICAL_DRIFT_AND_DEGRADATION",
        "P2_CRITICAL_DRIFT",
    ]
    assert report["action"] == "RETRAIN_PROMOTED"
    assert report["promoted_version"] is not None
    assert "festival" in report["root_cause"].lower() or "wl_position" in report["root_cause"].lower()

    # Active model in registry must now be updated
    active = get_active(custom_path=supervisor_env["reg_path"])
    assert active["version"] == report["promoted_version"]

    # Verify audit trail contains complete incident history
    all_events = get_events(db_path=supervisor_env["audit_path"])
    event_types = [e["event_type"] for e in all_events]
    assert "DRIFT_ALERT" in event_types
    assert "SUPERVISOR_DECISION" in event_types


def test_supervisor_escalates_when_retraining_rejected(supervisor_env):
    drifted_df = pd.read_csv(DATA_DIR / "drifted.csv")

    # Mock retrain_if_needed returning a rejected candidate
    mock_retrain_result = {
        "action": "rejected",
        "promoted": False,
        "candidate_version": "v2_bad",
        "active_version": "v1_sup_test",
        "candidate_holdout_metrics": {"auc": 0.65},
        "active_holdout_metrics": {"auc": 0.75},
        "holdout_samples_count": 900,
        "reason": "Candidate AUC fell below tolerance.",
    }

    with patch("app.supervisor.retrain_if_needed", return_value=mock_retrain_result):
        report = run_supervisor_cycle(
            current_batch_df=drifted_df,
            custom_registry_path=supervisor_env["reg_path"],
            custom_db_path=supervisor_env["audit_path"],
            custom_baseline_df=supervisor_env["baseline_df"],
        )

    assert report["action"] == "ESCALATE_TO_HUMAN"
    assert "escalated" in report["resolution"].lower()

    # Active version should NOT have changed
    active = get_active(custom_path=supervisor_env["reg_path"])
    assert active["version"] == "v1_sup_test"


def test_supervisor_halts_on_corrupted_batch(supervisor_env):
    bad_batch = pd.DataFrame([{"invalid_col": 123}])

    report = run_supervisor_cycle(
        current_batch_df=bad_batch,
        custom_registry_path=supervisor_env["reg_path"],
        custom_db_path=supervisor_env["audit_path"],
        custom_baseline_df=supervisor_env["baseline_df"],
    )

    assert report["status"] == "BLOCKED"
    assert report["action"] == "HALT_BATCH"
    assert "data-quality guardrail" in report["root_cause"].lower()

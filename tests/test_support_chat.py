"""
tests/test_support_chat.py
Unit and governance tests for Support Chatbot module.
Enforces test isolation: all tests use temporary directories for models,
registry files, and audit databases so production assets are never modified.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.config import (
    CHAT_EVAL_PATH,
    CHAT_SIMILARITY_THRESHOLD,
    FAQ_DIR,
    MOCK_PNR_PATH,
)
from app.governance.audit import get_events, init_audit_db
from app.governance.drift import detect_drift
from app.modules.support_chat import (
    detect_script_language,
    mask_pnr_in_query,
    support_service,
)
from app.supervisor import run_supervisor_cycle
from data.generate_chat_data import generate_chat_telemetry


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Provide isolated paths for models, registry, and audit db."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.modules.support_chat.MODELS_DIR", models_dir)

    db_path = tmp_path / "test_audit.db"
    init_audit_db(db_path)

    reg_path = tmp_path / "test_registry.json"
    return {"models_dir": models_dir, "db_path": db_path, "reg_path": reg_path}


def test_multilingual_faq_indexing_and_script_detection():
    """Verify FAQ knowledge base files exist and Unicode script detection works accurately."""
    assert FAQ_DIR.exists()
    faq_files = list(FAQ_DIR.glob("*.md"))
    assert len(faq_files) >= 12

    # Script detection: Devanagari vs Latin
    assert detect_script_language("What is the refund rule on ticket cancellation?") == "en"
    assert detect_script_language("कन्फर्म टिकट कैंसिल करने पर कितना रिफंड मिलता है?") == "hi"
    assert detect_script_language("पीएनआर स्टेटस क्या है?") == "hi"
    assert detect_script_language("Tatkal booking timings for AC") == "en"


def test_pnr_masking_in_audit_log(isolated_env):
    """
    STRICT GOVERNANCE PRIVACY TEST:
    Verify that any 10-digit number in the question is masked before writing to the audit log.
    The raw 10-digit number must NEVER appear in the audit database.
    """
    raw_query = "Please check status for my booking PNR 9823471029 immediately"
    masked = mask_pnr_in_query(raw_query)

    assert "9823471029" not in masked
    assert "******1029" in masked

    # Process query through support service
    res = support_service.predict(raw_query, custom_db_path=isolated_env["db_path"])
    assert res["is_grounded"] is True
    assert "9823471029" not in res["masked_query"]
    assert "******1029" in res["masked_query"]

    # Verify immutable audit log record
    events = get_events(
        event_type="CHAT_INTERACTION",
        model_name="support_chatbot",
        db_path=isolated_env["db_path"],
    )
    assert len(events) == 1
    logged_query = events[0]["details"]["masked_query"]
    assert "9823471029" not in logged_query
    assert "******1029" in logged_query


def test_pnr_lookup_and_missing(isolated_env):
    """Verify 10-digit PNR lookup in synthetic database and clean missing PNR response."""
    # 1. Valid mock PNR
    res_valid = support_service.predict("Check PNR 1234567890", custom_db_path=isolated_env["db_path"])
    assert res_valid["is_grounded"] is True
    assert "Aarav Sharma" in res_valid["reply"]
    assert "Mumbai Rajdhani" in res_valid["reply"]
    assert "CNF" in res_valid["reply"]

    # 2. Missing mock PNR
    res_missing = support_service.predict("PNR 9999999999 status", custom_db_path=isolated_env["db_path"])
    assert res_missing["is_grounded"] is True
    assert "not found" in res_missing["reply"].lower() or "flushed" in res_missing["reply"].lower()


def test_grounding_check_and_hallucination_prevention(isolated_env):
    """
    Verify demonstrable grounding check:
    - In-scope questions match verified FAQ documentation.
    - Out-of-scope questions MUST trigger the 'not sure' response and human escalation.
    """
    # 1. In-scope English question
    res_in_en = support_service.predict(
        "What time does Tatkal booking open for AC coaches?", custom_db_path=isolated_env["db_path"]
    )
    assert res_in_en["is_grounded"] is True
    assert res_in_en["is_escalated"] is False
    assert "tatkal_rules_en.md" in res_in_en["source"]
    assert res_in_en["similarity"] >= CHAT_SIMILARITY_THRESHOLD

    # 2. In-scope Hindi question
    res_in_hi = support_service.predict(
        "कन्फर्म टिकट कैंसिल करने पर कितना रिफंड मिलता है?", custom_db_path=isolated_env["db_path"]
    )
    assert res_in_hi["is_grounded"] is True
    assert res_in_hi["is_escalated"] is False
    assert "refunds_hi.md" in res_in_hi["source"]
    assert res_in_hi["language"] == "hi"

    # 3. Out-of-scope question (Flight booking) -> Hallucination check
    res_oos1 = support_service.predict(
        "Can I book a domestic flight ticket to Bangalore on this site?", custom_db_path=isolated_env["db_path"]
    )
    assert res_oos1["is_grounded"] is False
    assert res_oos1["is_escalated"] is True
    assert "not sure" in res_oos1["reply"].lower()
    assert "connect you with a railway support executive" in res_oos1["reply"].lower()

    # 4. Out-of-scope Hindi question (Hotel booking)
    res_oos2 = support_service.predict(
        "होटल रूम बुकिंग के लिए कौन सा कूपन कोड उपलब्ध है?", custom_db_path=isolated_env["db_path"]
    )
    assert res_oos2["is_grounded"] is False
    assert res_oos2["is_escalated"] is True
    assert "पूरी जानकारी नहीं है" in res_oos2["reply"]


def test_evaluation_on_test_split():
    """Verify retrieval accuracy and rejection metrics on the held-out test split of chat_eval.csv."""
    assert CHAT_EVAL_PATH.exists()
    eval_metrics = support_service.evaluate_test_split()

    assert eval_metrics["test_samples"] == 20
    # Required threshold accuracy on test set
    assert eval_metrics["accuracy"] >= 0.85
    # Demonstrable hallucination check: all out-of-scope test questions rejected
    assert eval_metrics["out_of_scope_rejection_rate"] == 1.0
    # In-scope wrong answer rate < 15%
    assert eval_metrics["wrong_answer_rate"] <= 0.15


def test_feedback_audit_logging(isolated_env):
    """Verify plain text feedback (Helpful / Not helpful) audit logging."""
    support_service.log_feedback(
        masked_query="What is the refund rule?",
        is_helpful=True,
        custom_db_path=isolated_env["db_path"],
    )
    support_service.log_feedback(
        masked_query="Check PNR ******1029",
        is_helpful=False,
        custom_db_path=isolated_env["db_path"],
    )

    events = get_events(
        event_type="CHAT_FEEDBACK",
        model_name="support_chatbot",
        db_path=isolated_env["db_path"],
    )
    assert len(events) == 2
    assert events[0]["details"]["helpful"] is False
    assert events[1]["details"]["helpful"] is True


def test_chat_drift_and_supervisor_cycle(isolated_env):
    """Verify covariate drift detection on telemetry batches and supervisor cycle execution."""
    base_df = generate_chat_telemetry(n_samples=200, is_drifted=False, random_seed=42)
    drift_df = generate_chat_telemetry(n_samples=200, is_drifted=True, random_seed=101)

    # 1. Covariate drift
    drift_report = detect_drift(
        base_df,
        drift_df,
        feature_columns=support_service.feature_columns,
        categorical_features=support_service.categorical_features,
        numeric_features=support_service.numeric_features,
    )
    assert drift_report["overall_drift_detected"] is True

    # 2. Supervisor governance cycle
    report = run_supervisor_cycle(
        drift_df,
        service=support_service,
        custom_registry_path=isolated_env["reg_path"],
        custom_db_path=isolated_env["db_path"],
        custom_baseline_df=base_df,
    )
    assert report["model_name"] == "support_chatbot"
    assert "drift_summary" in report
    assert "action" in report


def test_chat_trip_planning_integration(isolated_env):
    """Verify conversational trip planning queries route to SmartPlanner and return grounded itineraries."""
    res = support_service.predict(
        "Pune to Delhi tomorrow under ₹1500, reach before 10 AM",
        custom_db_path=isolated_env["db_path"],
    )
    assert res["is_grounded"] is True
    assert "Trip Planner" in res["badge"]
    assert "Best Train Found" in res["reply"]
    assert "trip_data" in res
    assert res["trip_data"]["best_itinerary"] is not None
    assert "Pune" in res["reply"] or "Delhi" in res["reply"]


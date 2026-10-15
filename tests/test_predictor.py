"""
tests/test_predictor.py
Tests for predictor module:
- Model training, evaluation metrics, and artifact persistence.
- Output schema: probability, status, confidence, and top SHAP factors.
- SHAP explanations: verify no percentage claims, adherence to strength labels, and log-odds semantics.
- Domain logic sanity: RAC / low WL tickets have higher confirmation probability than high WL tickets.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.config import DATA_DIR, MODELS_DIR
from app.modules.ticket_predictor import load_model, predict, train


@pytest.fixture(scope="module")
def trained_model():
    """Train a test version on baseline data and make it available across tests."""
    baseline_path = DATA_DIR / "baseline.csv"
    assert baseline_path.exists(), "baseline.csv must exist before running predictor tests"
    df = pd.read_csv(baseline_path)

    res = train(df, version="test_v1")
    bundle, meta = load_model("test_v1")
    yield bundle, meta, res

    # Clean up test artifacts
    for f in [
        MODELS_DIR / "ticket_predictor_test_v1.joblib",
        MODELS_DIR / "ticket_predictor_test_v1_meta.joblib",
        MODELS_DIR / "test_v1.joblib",
        MODELS_DIR / "test_v1_meta.joblib",
    ]:
        if f.exists():
            f.unlink()


def test_model_training_metrics(trained_model):
    _, meta, res = trained_model
    metrics = res["metrics"]

    assert "auc" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics

    # Target AUC should be roughly 0.80 - 0.88
    assert 0.78 <= metrics["auc"] <= 0.92
    assert metrics["accuracy"] >= 0.70
    assert metrics["f1"] >= 0.70

    # Verify feature stats were extracted
    assert "feature_stats" in res
    assert "numeric" in res["feature_stats"]
    assert "categorical" in res["feature_stats"]


def test_predict_schema_and_confidence(trained_model):
    bundle, _, _ = trained_model
    sample_ticket = {
        "booking_id": 10001,
        "travel_class": "3A",
        "booking_status": "WL",
        "wl_position": 25,
        "days_before_journey": 45,
        "day_of_week": 3,
        "month": 6,
        "is_festival_season": 0,
        "route_popularity": 0.45,
        "train_type": "Superfast",
    }

    result = predict(sample_ticket, bundle)

    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_status"] in [
        "Likely Confirmed",
        "Uncertain",
        "Likely Not Confirmed",
    ]
    assert result["confidence"] in ["High", "Medium", "Low"]
    assert len(result["top_factors"]) == 3


def test_shap_explanations_wording_and_no_percentage(trained_model):
    bundle, _, _ = trained_model
    sample_ticket = {
        "booking_id": 10002,
        "travel_class": "SL",
        "booking_status": "WL",
        "wl_position": 95,
        "days_before_journey": 5,
        "day_of_week": 5,
        "month": 11,
        "is_festival_season": 1,
        "route_popularity": 0.85,
        "train_type": "Rajdhani",
    }

    result = predict(sample_ticket, bundle)
    top_factors = result["top_factors"]

    # Verify ranking by absolute SHAP magnitude
    abs_shaps = [abs(f["shap_value"]) for f in top_factors]
    assert abs_shaps == sorted(abs_shaps, reverse=True)

    for factor in top_factors:
        assert factor["strength"] in ["strongly", "moderately", "slightly"]
        assert factor["direction"] in [
            "increased the chance of confirmation",
            "reduced the chance of confirmation",
        ]
        explanation = factor["explanation"]
        assert isinstance(explanation, str)
        # Requirement: Do not state exact percentages in SHAP explanations
        assert "%" not in explanation, f"Percentage sign found in explanation: {explanation}"
        assert factor["strength"] in explanation


def test_domain_prediction_contrast(trained_model):
    bundle, _, _ = trained_model

    # Ticket A: Favourable (RAC 5, 60 days before journey, Sleeper, off-peak)
    ticket_favourable = {
        "travel_class": "SL",
        "booking_status": "RAC",
        "wl_position": 5,
        "days_before_journey": 60,
        "day_of_week": 2,
        "month": 4,
        "is_festival_season": 0,
        "route_popularity": 0.30,
        "train_type": "Express",
    }

    # Ticket B: Unfavourable (WL 150, 2 days before journey, 1A, peak festival rush)
    ticket_unfavourable = {
        "travel_class": "1A",
        "booking_status": "WL",
        "wl_position": 150,
        "days_before_journey": 2,
        "day_of_week": 5,
        "month": 10,
        "is_festival_season": 1,
        "route_popularity": 0.90,
        "train_type": "Rajdhani",
    }

    res_a = predict(ticket_favourable, bundle)
    res_b = predict(ticket_unfavourable, bundle)

    assert res_a["probability"] > res_b["probability"]
    assert res_a["probability"] >= 0.70
    assert res_b["probability"] <= 0.35

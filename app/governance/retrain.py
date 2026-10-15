"""
app/governance/retrain.py
Autonomous Model Retraining & Validation Pipeline for RailGov MVP.
Orchestrates candidate model retraining upon drift or performance alerts.
Enforces holdout data isolation: candidate models are evaluated on recent holdouts
that are strictly excluded from their training set.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import (
    DATA_DIR,
    RETRAIN_PROMOTION_TOLERANCE,
    TARGET_COLUMN,
)
from app.governance.audit import log_event
from app.governance.monitor import evaluate_performance
from app.governance.registry import get_active, list_versions, promote, register
from app.predictor import load_model, train


def _determine_next_version() -> str:
    """Determine the next sequential version tag (e.g. v2, v3)."""
    versions = list_versions()
    max_num = 0
    for v in versions:
        ver_str = v.get("version", "")
        if ver_str.startswith("v") and ver_str[1:].isdigit():
            max_num = max(max_num, int(ver_str[1:]))
    return f"v{max_num + 1}"


def retrain_if_needed(
    trigger_reason: str,
    current_df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    custom_registry_path: Optional[Path] = None,
    custom_db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute autonomous retraining pipeline.

    Workflow:
    1. Splits newly labelled current data into training (70%) and validation holdout (30%).
       Constraint: The 30% holdout is strictly excluded from candidate training.
    2. Combines baseline operations with the 70% current data slice.
    3. Trains candidate model version (e.g. v2).
    4. Evaluates active vs. candidate models on the unseen 30% holdout.
    5. Promotes candidate if holdout AUC >= active AUC - tolerance; otherwise marks rejected.
    6. Logs all audit decisions.

    Args:
        trigger_reason: Explanation of what caused retraining (e.g. "critical drift in festival season").
        current_df: Labelled recent production data.
        baseline_df: Optional baseline data; loaded from data/baseline.csv if None.
        custom_registry_path: Optional registry path for testing.
        custom_db_path: Optional audit db path for testing.

    Returns:
        Retraining report dictionary.
    """
    if baseline_df is None:
        baseline_path = DATA_DIR / "baseline.csv"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline data not found at {baseline_path}")
        baseline_df = pd.read_csv(baseline_path)

    active_rec = get_active(custom_path=custom_registry_path)
    active_version = active_rec["version"] if active_rec else None

    candidate_version = _determine_next_version()

    # Log retraining started
    log_event(
        event_type="RETRAIN_STARTED",
        actor="retrain_pipeline",
        details={
            "trigger_reason": trigger_reason,
            "candidate_version": candidate_version,
            "active_version": active_version,
            "new_samples_count": len(current_df),
        },
        model_version=active_version,
        db_path=custom_db_path,
    )

    # 1. Split recent data into training (70%) and holdout (30%)
    # This ensures the validation holdout is strictly unseen by the candidate model
    current_train, current_holdout = train_test_split(
        current_df,
        test_size=0.30,
        random_state=42,
        stratify=current_df[TARGET_COLUMN],
    )

    # 2. Combine baseline data with recent training slice
    combined_train = pd.concat([baseline_df, current_train], ignore_index=True)

    # 3. Train candidate model
    train_res = train(combined_train, version=candidate_version)

    # 4. Evaluate candidate model on recent unseen holdout
    candidate_bundle, _ = load_model(candidate_version)
    candidate_holdout_eval = evaluate_performance(candidate_bundle, current_holdout)
    candidate_auc = candidate_holdout_eval["metrics"]["auc"]

    # 5. Evaluate current active model on the same unseen holdout (if active exists)
    if active_version:
        active_bundle, _ = load_model(active_version)
        active_holdout_eval = evaluate_performance(active_bundle, current_holdout)
        active_auc = active_holdout_eval["metrics"]["auc"]
    else:
        active_auc = 0.0
        active_holdout_eval = {"metrics": {"auc": 0.0}}

    # 6. Promotion decision
    min_required_auc = active_auc - RETRAIN_PROMOTION_TOLERANCE
    is_promoted = candidate_auc >= min_required_auc

    if is_promoted:
        register(
            version=candidate_version,
            metrics=candidate_holdout_eval["metrics"],
            training_data_desc=f"Baseline + 70% current data ({trigger_reason})",
            status="active",
            custom_path=custom_registry_path,
        )
        log_event(
            event_type="MODEL_PROMOTED",
            actor="retrain_pipeline",
            details={
                "candidate_version": candidate_version,
                "previous_version": active_version,
                "candidate_holdout_auc": candidate_auc,
                "active_holdout_auc": active_auc,
                "tolerance": RETRAIN_PROMOTION_TOLERANCE,
                "holdout_samples": len(current_holdout),
            },
            model_version=candidate_version,
            db_path=custom_db_path,
        )
        action = "promoted"
        reason = (
            f"Candidate model {candidate_version} holdout AUC ({candidate_auc}) "
            f"met or exceeded active {active_version} AUC ({active_auc}) on recent unseen holdout."
        )
    else:
        register(
            version=candidate_version,
            metrics=candidate_holdout_eval["metrics"],
            training_data_desc=f"Baseline + 70% current data ({trigger_reason})",
            status="rejected",
            custom_path=custom_registry_path,
        )
        log_event(
            event_type="MODEL_REJECTED",
            actor="retrain_pipeline",
            details={
                "candidate_version": candidate_version,
                "active_version": active_version,
                "candidate_holdout_auc": candidate_auc,
                "active_holdout_auc": active_auc,
                "tolerance": RETRAIN_PROMOTION_TOLERANCE,
                "holdout_samples": len(current_holdout),
            },
            model_version=active_version,
            db_path=custom_db_path,
        )
        action = "rejected"
        reason = (
            f"Candidate model {candidate_version} holdout AUC ({candidate_auc}) "
            f"did not meet required threshold ({min_required_auc}) compared to active model ({active_auc})."
        )

    return {
        "action": action,
        "promoted": is_promoted,
        "candidate_version": candidate_version,
        "active_version": candidate_version if is_promoted else active_version,
        "candidate_holdout_metrics": candidate_holdout_eval["metrics"],
        "active_holdout_metrics": active_holdout_eval["metrics"],
        "holdout_samples_count": len(current_holdout),
        "reason": reason,
    }

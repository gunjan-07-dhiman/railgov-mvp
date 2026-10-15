"""
app/governance/retrain.py
Autonomous Model Retraining & Validation Pipeline for RailGov MVP.
Orchestrates candidate model retraining upon drift or performance alerts.
Enforces holdout data isolation across governed model services.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from app.base import ModelService, get_service
from app.config import RETRAIN_PROMOTION_TOLERANCE
from app.governance.audit import log_event
from app.governance.registry import get_active, list_versions, register


def _determine_next_version(model_name: str, custom_path: Optional[Path] = None) -> str:
    """Determine the next sequential version tag for a model (e.g. v2, v3)."""
    versions = list_versions(model_name=model_name, custom_path=custom_path)
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
    service: Optional[ModelService] = None,
    custom_registry_path: Optional[Path] = None,
    custom_db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute autonomous retraining pipeline with unseen holdout isolation."""
    if service is None:
        service = get_service("ticket_predictor")

    model_name = service.model_name
    target_col = service.target_column

    if baseline_df is None:
        baseline_df = service.load_baseline_df()

    active_rec = get_active(model_name=model_name, custom_path=custom_registry_path)
    active_version = active_rec["version"] if active_rec else None
    candidate_version = _determine_next_version(model_name, custom_path=custom_registry_path)

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
        model_name=model_name,
        db_path=custom_db_path,
    )

    # 1. Split recent data: 70% train, 30% unseen holdout
    stratify_col = None
    if target_col in current_df.columns:
        counts = current_df[target_col].value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            stratify_col = current_df[target_col]

    current_train, current_holdout = train_test_split(
        current_df, test_size=0.30, random_state=42, stratify=stratify_col
    )

    # 2. Combine baseline data with recent training slice
    combined_train = pd.concat([baseline_df, current_train], ignore_index=True)

    # 3. Train candidate model
    service.train(combined_train, version=candidate_version)

    # 4. Evaluate candidate on unseen holdout
    candidate_bundle, _ = service.load_model(candidate_version)
    candidate_holdout_eval = service.evaluate_performance(candidate_bundle, current_holdout)
    candidate_auc = candidate_holdout_eval["metrics"]["auc"]

    # 5. Evaluate current active model on same holdout
    if active_version:
        active_bundle, _ = service.load_model(active_version)
        active_holdout_eval = service.evaluate_performance(active_bundle, current_holdout)
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
            model_name=model_name,
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
            model_name=model_name,
            db_path=custom_db_path,
        )
        action = "promoted"
        reason = f"Candidate model {candidate_version} holdout AUC ({candidate_auc}) met target (>= {min_required_auc})."
    else:
        register(
            version=candidate_version,
            metrics=candidate_holdout_eval["metrics"],
            training_data_desc=f"Baseline + 70% current data ({trigger_reason})",
            status="rejected",
            model_name=model_name,
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
            },
            model_version=active_version,
            model_name=model_name,
            db_path=custom_db_path,
        )
        action = "rejected"
        reason = f"Candidate model {candidate_version} holdout AUC ({candidate_auc}) below threshold ({min_required_auc})."

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

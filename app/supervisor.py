"""
app/supervisor.py
Autonomous AI Supervisor Agent for RailGov MVP.
Orchestrates governance controls for governed model services:
1. Validates batch data quality.
2. Ingests statistical drift and performance telemetry.
3. Prioritizes alerts.
4. Performs automated Root Cause Analysis (RCA).
5. Executes governance interventions (RETRAIN, MONITOR, ESCALATE_TO_HUMAN).
6. Records plain-English audit trails under SUPERVISOR_DECISION.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from app.base import ModelService, get_service
from app.config import AUC_DEGRADATION_THRESHOLD
from app.governance.audit import log_event
from app.governance.drift import detect_drift
from app.governance.registry import get_active, register
from app.governance.retrain import retrain_if_needed


def run_supervisor_cycle(
    current_batch_df: pd.DataFrame,
    service: Optional[ModelService] = None,
    custom_registry_path: Optional[Path] = None,
    custom_db_path: Optional[Path] = None,
    custom_baseline_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Execute an autonomous governance cycle across quality, drift, and performance."""
    if service is None:
        service = get_service("ticket_predictor")

    model_name = service.model_name
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Data Quality Guardrail Check
    is_valid, quality_issues = service.validate_batch(current_batch_df)
    if not is_valid:
        root_cause = f"Batch rejected by data-quality guardrail: {'; '.join(quality_issues[:3])}"
        report = {
            "timestamp": timestamp,
            "status": "BLOCKED",
            "priority": "P0_DATA_CORRUPTION",
            "action": "HALT_BATCH",
            "model_name": model_name,
            "root_cause": root_cause,
            "resolution": "Batch ingestion suspended. Upstream data pipeline alerted.",
            "data_quality_issues": quality_issues,
            "active_version": None,
        }
        log_event(
            event_type="DATA_QUALITY_BLOCK",
            actor="supervisor_agent",
            details=report,
            model_name=model_name,
            db_path=custom_db_path,
        )
        log_event(
            event_type="SUPERVISOR_DECISION",
            actor="supervisor_agent",
            details=report,
            model_name=model_name,
            db_path=custom_db_path,
        )
        return report

    # 2. Retrieve Active Model
    active_rec = get_active(model_name=model_name, custom_path=custom_registry_path)
    if not active_rec:
        baseline_df = custom_baseline_df if custom_baseline_df is not None else service.load_baseline_df()
        train_res = service.train(baseline_df, version="v1")
        register(
            version="v1",
            metrics=train_res["metrics"],
            training_data_desc="Bootstrap baseline training",
            status="active",
            model_name=model_name,
            custom_path=custom_registry_path,
        )
        active_rec = get_active(model_name=model_name, custom_path=custom_registry_path)

    active_version = active_rec["version"]
    active_bundle, active_meta = service.load_model(active_version)
    baseline_auc = active_meta.get("metrics", {}).get("auc", 0.85)

    # 3. Statistical Drift Assessment
    baseline_df = custom_baseline_df if custom_baseline_df is not None else service.load_baseline_df()
    drift_report = detect_drift(
        baseline_df,
        current_batch_df,
        feature_columns=service.feature_columns,
        categorical_features=service.categorical_features,
        numeric_features=service.numeric_features,
    )

    if drift_report["overall_drift_detected"]:
        log_event(
            event_type="DRIFT_ALERT",
            actor="supervisor_agent",
            details={
                "overall_severity": drift_report["overall_severity"],
                "critical_features": drift_report["critical_features"],
                "top_drifted": drift_report["top_drifted_features"][:3],
            },
            model_version=active_version,
            model_name=model_name,
            db_path=custom_db_path,
        )

    # 4. Performance Monitoring Assessment (if ground truth labels present)
    has_labels = service.target_column in current_batch_df.columns
    perf_report = None
    is_degraded = False
    auc_drop = 0.0

    if has_labels:
        perf_report = service.evaluate_performance(
            active_bundle, current_batch_df, baseline_metric=baseline_auc
        )
        is_degraded = perf_report.get("degradation_detected", False)
        auc_drop = perf_report.get("auc_drop", 0.0)

        if is_degraded:
            log_event(
                event_type="PERFORMANCE_ALERT",
                actor="supervisor_agent",
                details={
                    "baseline_auc": baseline_auc,
                    "current_auc": perf_report["metrics"]["auc"],
                    "auc_drop": auc_drop,
                    "threshold": AUC_DEGRADATION_THRESHOLD,
                },
                model_version=active_version,
                model_name=model_name,
                db_path=custom_db_path,
            )

    # 5. Alert Prioritization & RCA
    is_critical_drift = drift_report["overall_severity"] == "critical"
    is_warning_drift = drift_report["overall_severity"] == "warning"

    top_features = [
        f"{item['feature']} (PSI={item['psi']:.2f})"
        for item in drift_report["top_drifted_features"][:2]
    ]
    top_features_str = ", ".join(top_features) if top_features else "none"

    if is_critical_drift and is_degraded:
        priority = "P1_CRITICAL_DRIFT_AND_DEGRADATION"
        action = "RETRAIN"
        root_cause = (
            f"Severe covariate shift in {top_features_str}. "
            f"Active model '{active_version}' AUC dropped from {baseline_auc:.3f} to "
            f"{perf_report['metrics']['auc']:.3f} (drop of {auc_drop:.3f} > {AUC_DEGRADATION_THRESHOLD})."
        )
    elif is_critical_drift:
        priority = "P2_CRITICAL_DRIFT"
        action = "RETRAIN"
        root_cause = f"Critical statistical drift detected in {top_features_str}. Proactive retraining triggered."
    elif is_degraded:
        priority = "P3_PERFORMANCE_DROP"
        action = "RETRAIN"
        root_cause = f"Model degradation observed: AUC fell by {auc_drop:.3f} on recent batch."
    elif is_warning_drift:
        priority = "P4_DRIFT_WARNING"
        action = "MONITOR"
        root_cause = f"Mild distribution shift in {top_features_str} within warning bounds."
    else:
        priority = "P5_HEALTHY"
        action = "MONITOR"
        root_cause = "All data quality, stability, and accuracy metrics are healthy."

    # 6. Action Execution
    resolution = ""
    candidate_version = None

    if action == "RETRAIN":
        retrain_result = retrain_if_needed(
            trigger_reason=root_cause,
            current_df=current_batch_df,
            baseline_df=baseline_df,
            service=service,
            custom_registry_path=custom_registry_path,
            custom_db_path=custom_db_path,
        )
        candidate_version = retrain_result["candidate_version"]

        if retrain_result["promoted"]:
            action = "RETRAIN_PROMOTED"
            resolution = f"Candidate model '{candidate_version}' promoted to active. Reason: {retrain_result['reason']}"
        else:
            action = "ESCALATE_TO_HUMAN"
            resolution = f"Candidate model '{candidate_version}' rejected on holdout. Retaining '{active_version}'. Escalated to engineering."
    else:
        resolution = "Operational state normal. Telemetry continuously monitored."

    # 7. Assemble Structured Report
    report = {
        "timestamp": timestamp,
        "priority": priority,
        "action": action,
        "model_name": model_name,
        "active_version": active_version,
        "promoted_version": candidate_version if action == "RETRAIN_PROMOTED" else None,
        "root_cause": root_cause,
        "resolution": resolution,
        "drift_summary": {
            "overall_severity": drift_report["overall_severity"],
            "critical_count": drift_report["critical_count"],
            "top_drifted": drift_report["top_drifted_features"][:3],
        },
        "performance_summary": perf_report["metrics"] if perf_report else None,
    }

    # 8. Log Final Supervisor Decision
    log_event(
        event_type="SUPERVISOR_DECISION",
        actor="supervisor_agent",
        details=report,
        model_version=report["promoted_version"] or active_version,
        model_name=model_name,
        db_path=custom_db_path,
    )

    return report

"""
app/supervisor.py
Autonomous Rule-Based AI Supervisor Agent for RailGov MVP.
Orchestrates governance controls without external LLM dependencies:
1. Validates batch data quality.
2. Ingests statistical drift and performance telemetry.
3. Prioritizes alerts (Critical Drift + Degradation > Drift > Warning).
4. Performs automated Root Cause Analysis (RCA).
5. Executes governance interventions (RETRAIN, ROLLBACK, MONITOR, ESCALATE_TO_HUMAN).
6. Records plain-English audit trails under SUPERVISOR_DECISION.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from app.config import (
    AUC_DEGRADATION_THRESHOLD,
    DATA_DIR,
    TARGET_COLUMN,
)
from app.governance.audit import log_event
from app.governance.data_quality import validate_batch
from app.governance.drift import detect_drift
from app.governance.monitor import evaluate_performance
from app.governance.registry import get_active, rollback
from app.governance.retrain import retrain_if_needed
from app.predictor import load_model, train


def generate_llm_summary_hook(report: Dict[str, Any]) -> str:
    """
    Extensibility hook for pluggable LLM summarization.
    Allows future integration with an LLM without changing the core rule-based supervisor contract.
    """
    return (
        f"Automated Supervisor Report ({report.get('timestamp')}):\n"
        f"Alert Level: {report.get('priority')}\n"
        f"Diagnosis: {report.get('root_cause')}\n"
        f"Action: {report.get('action')}\n"
        f"Resolution: {report.get('resolution')}"
    )


def run_supervisor_cycle(
    current_batch_df: pd.DataFrame,
    custom_registry_path: Optional[Path] = None,
    custom_db_path: Optional[Path] = None,
    custom_baseline_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Execute an autonomous governance cycle across data quality, drift, and performance.

    Args:
        current_batch_df: Ingested production batch DataFrame.
        custom_registry_path: Optional registry file path (for unit testing).
        custom_db_path: Optional audit database path (for unit testing).
        custom_baseline_df: Optional baseline reference DataFrame.

    Returns:
        Structured supervisor incident report.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Ingestion Guardrail: Data Quality Check
    is_valid, quality_issues = validate_batch(current_batch_df)
    if not is_valid:
        root_cause = f"Batch rejected by data-quality guardrail: {'; '.join(quality_issues[:3])}"
        report = {
            "timestamp": timestamp,
            "status": "BLOCKED",
            "priority": "P0_DATA_CORRUPTION",
            "action": "HALT_BATCH",
            "root_cause": root_cause,
            "resolution": "Batch ingestion suspended. Upstream data pipeline alerted.",
            "data_quality_issues": quality_issues,
            "active_version": None,
        }
        log_event(
            event_type="DATA_QUALITY_BLOCK",
            actor="supervisor_agent",
            details=report,
            db_path=custom_db_path,
        )
        log_event(
            event_type="SUPERVISOR_DECISION",
            actor="supervisor_agent",
            details=report,
            db_path=custom_db_path,
        )
        return report

    # 2. Retrieve Active Model
    active_rec = get_active(custom_path=custom_registry_path)
    if not active_rec:
        # Bootstrap v1 if not present
        baseline_path = DATA_DIR / "baseline.csv"
        baseline_df = (
            custom_baseline_df
            if custom_baseline_df is not None
            else pd.read_csv(baseline_path)
        )
        train_res = train(baseline_df, version="v1")
        from app.governance.registry import register

        register(
            version="v1",
            metrics=train_res["metrics"],
            training_data_desc="Bootstrap baseline training",
            status="active",
            custom_path=custom_registry_path,
        )
        active_rec = get_active(custom_path=custom_registry_path)

    active_version = active_rec["version"]
    active_bundle, active_meta = load_model(active_version)
    baseline_auc = active_meta.get("metrics", {}).get("auc", 0.85)

    # 3. Statistical Drift Assessment
    baseline_df = (
        custom_baseline_df
        if custom_baseline_df is not None
        else pd.read_csv(DATA_DIR / "baseline.csv")
    )
    drift_report = detect_drift(baseline_df, current_batch_df)

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
            db_path=custom_db_path,
        )

    # 4. Performance Monitoring Assessment (if batch has ground truth labels)
    has_labels = TARGET_COLUMN in current_batch_df.columns
    perf_report = None
    is_degraded = False
    auc_drop = 0.0

    if has_labels:
        perf_report = evaluate_performance(
            active_bundle, current_batch_df, baseline_auc=baseline_auc
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
                db_path=custom_db_path,
            )

    # 5. Alert Prioritization & Root Cause Analysis
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
            f"Active model '{active_version}' AUC degraded from {baseline_auc:.3f} to "
            f"{perf_report['metrics']['auc']:.3f} (drop of {auc_drop:.3f} > {AUC_DEGRADATION_THRESHOLD})."
        )
    elif is_critical_drift:
        priority = "P2_CRITICAL_DRIFT"
        action = "RETRAIN"
        root_cause = (
            f"Critical statistical drift detected in {top_features_str}. "
            f"Proactive retraining initiated before consumer performance degrades."
        )
    elif is_degraded:
        priority = "P3_PERFORMANCE_DROP"
        action = "RETRAIN"
        root_cause = (
            f"Model degradation observed: AUC fell by {auc_drop:.3f} on recent batch "
            f"exceeding stability threshold {AUC_DEGRADATION_THRESHOLD}."
        )
    elif is_warning_drift:
        priority = "P4_DRIFT_WARNING"
        action = "MONITOR"
        root_cause = (
            f"Mild distribution shift observed in {top_features_str} within warning bounds. "
            f"No retraining required."
        )
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
            custom_registry_path=custom_registry_path,
            custom_db_path=custom_db_path,
        )
        candidate_version = retrain_result["candidate_version"]

        if retrain_result["promoted"]:
            action = "RETRAIN_PROMOTED"
            resolution = (
                f"Candidate model '{candidate_version}' trained on combined data, "
                f"validated on holdout, and promoted to active. Reason: {retrain_result['reason']}"
            )
        else:
            action = "ESCALATE_TO_HUMAN"
            resolution = (
                f"Candidate model '{candidate_version}' was rejected during holdout validation. "
                f"Retaining existing active model '{active_version}'. "
                f"Escalated incident to ML engineering team."
            )
    else:
        resolution = "Operational state normal. Telemetry continuously monitored."

    # 7. Assemble Structured Report
    report = {
        "timestamp": timestamp,
        "priority": priority,
        "action": action,
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

    # Add optional LLM summary hook
    report["plain_english_summary"] = generate_llm_summary_hook(report)

    # 8. Log Final Supervisor Decision
    log_event(
        event_type="SUPERVISOR_DECISION",
        actor="supervisor_agent",
        details=report,
        model_version=report["promoted_version"] or active_version,
        db_path=custom_db_path,
    )

    return report

"""
app/governance/monitor.py
Continuous Performance Monitoring for RailGov MVP.
Evaluates model metrics on newly ground-truthed production batches
and flags performance degradation against baseline thresholds.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.config import (
    AUC_DEGRADATION_THRESHOLD,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def evaluate_performance(
    bundle: Dict[str, Any],
    labelled_df: pd.DataFrame,
    baseline_auc: Optional[float] = None,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate model performance on a labelled dataset with degradation check."""
    tgt = target_column if target_column is not None else TARGET_COLUMN
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS

    if tgt not in labelled_df.columns:
        raise ValueError(f"Dataset must contain target column '{tgt}' for performance evaluation.")

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    X = labelled_df[cols]
    y_true = labelled_df[tgt].astype(int).values

    if len(set(y_true)) < 2:
        return {
            "error": "Single class present in dataset. ROC-AUC cannot be calculated.",
            "sample_count": len(y_true),
            "degradation_detected": False,
        }

    X_trans = preprocessor.transform(X)
    y_prob = model.predict_proba(X_trans)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    current_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    metrics = {
        "auc": current_auc,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sample_count": len(y_true),
    }

    degradation_detected = False
    auc_drop = 0.0
    if baseline_auc is not None:
        auc_drop = round(float(baseline_auc - current_auc), 4)
        if auc_drop > AUC_DEGRADATION_THRESHOLD:
            degradation_detected = True

    return {
        "metrics": metrics,
        "baseline_auc": baseline_auc,
        "auc_drop": auc_drop,
        "degradation_detected": degradation_detected,
        "status": "degraded" if degradation_detected else "healthy",
    }

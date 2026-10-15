"""
app/governance/drift.py
Drift Detection Engine for RailGov MVP.
Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to detect covariate shift between reference and production data batches.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    KS_PVALUE_ALPHA,
    NUMERIC_FEATURES,
    PSI_CRITICAL_THRESHOLD,
    PSI_NUM_BINS,
    PSI_WARNING_THRESHOLD,
)


def compute_numeric_psi(
    reference: Union[np.ndarray, pd.Series],
    current: Union[np.ndarray, pd.Series],
    bins: int = PSI_NUM_BINS,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index for continuous numeric features."""
    ref_clean = np.asarray(reference, dtype=float)
    cur_clean = np.asarray(current, dtype=float)
    ref_clean = ref_clean[~np.isnan(ref_clean)]
    cur_clean = cur_clean[~np.isnan(cur_clean)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    bin_edges = np.unique(np.quantile(ref_clean, quantiles))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(ref_clean.min() - 1e-3, ref_clean.max() + 1e-3, bins + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (len(ref_clean) + len(ref_counts) * epsilon)
    cur_pct = (cur_counts + epsilon) / (len(cur_clean) + len(cur_counts) * epsilon)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return max(0.0, float(psi_val))


def compute_categorical_psi(
    reference: Union[List, pd.Series],
    current: Union[List, pd.Series],
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index for categorical features across shared categories."""
    ref_s = pd.Series(reference).dropna().astype(str)
    cur_s = pd.Series(current).dropna().astype(str)

    if len(ref_s) == 0 or len(cur_s) == 0:
        return 0.0

    all_categories = sorted(list(set(ref_s.unique()) | set(cur_s.unique())))
    k = len(all_categories)
    ref_counts = ref_s.value_counts()
    cur_counts = cur_s.value_counts()

    ref_arr = np.array([ref_counts.get(cat, 0) for cat in all_categories], dtype=float)
    cur_arr = np.array([cur_counts.get(cat, 0) for cat in all_categories], dtype=float)

    ref_pct = (ref_arr + epsilon) / (len(ref_s) + k * epsilon)
    cur_pct = (cur_arr + epsilon) / (len(cur_s) + k * epsilon)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return max(0.0, float(psi_val))


def compute_ks_test(
    reference: Union[np.ndarray, pd.Series],
    current: Union[np.ndarray, pd.Series],
) -> Tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov test for continuous numeric distribution shift."""
    ref_clean = np.asarray(reference, dtype=float)
    cur_clean = np.asarray(current, dtype=float)
    ref_clean = ref_clean[~np.isnan(ref_clean)]
    cur_clean = cur_clean[~np.isnan(cur_clean)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0, 1.0

    stat, p_val = ks_2samp(ref_clean, cur_clean)
    return float(stat), float(p_val)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute statistical drift analysis comparing current batch to baseline reference."""
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    cats = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    feature_metrics: Dict[str, Dict[str, Any]] = {}
    critical_features: List[str] = []
    warning_features: List[str] = []

    for col in cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        if col in cats:
            psi = compute_categorical_psi(reference_df[col], current_df[col])
            ks_stat, ks_p = None, None
        else:
            psi = compute_numeric_psi(reference_df[col], current_df[col])
            ks_stat, ks_p = compute_ks_test(reference_df[col], current_df[col])

        if psi >= PSI_CRITICAL_THRESHOLD:
            severity = "critical"
            critical_features.append(col)
        elif psi >= PSI_WARNING_THRESHOLD:
            severity = "warning"
            warning_features.append(col)
        else:
            severity = "none"

        feature_metrics[col] = {
            "psi": round(float(psi), 4),
            "ks_statistic": round(float(ks_stat), 4) if ks_stat is not None else None,
            "ks_pvalue": float(ks_p) if ks_p is not None else None,
            "ks_drift_flag": bool(ks_p < KS_PVALUE_ALPHA) if ks_p is not None else False,
            "severity": severity,
            "feature_type": "categorical" if col in cats else "numeric",
        }

    # System-level drift decision
    if len(critical_features) > 0:
        overall_severity = "critical"
        overall_drift_detected = True
    elif len(warning_features) >= 2:
        overall_severity = "warning"
        overall_drift_detected = True
    elif len(warning_features) == 1:
        overall_severity = "warning"
        overall_drift_detected = False
    else:
        overall_severity = "none"
        overall_drift_detected = False

    ranked = sorted(feature_metrics.items(), key=lambda item: item[1]["psi"], reverse=True)
    top_drifted = [
        {"feature": k, "psi": v["psi"], "severity": v["severity"]}
        for k, v in ranked
        if v["severity"] != "none"
    ]

    return {
        "overall_drift_detected": overall_drift_detected,
        "overall_severity": overall_severity,
        "critical_count": len(critical_features),
        "warning_count": len(warning_features),
        "critical_features": critical_features,
        "warning_features": warning_features,
        "top_drifted_features": top_drifted,
        "feature_metrics": feature_metrics,
        "sample_size": len(current_df),
    }

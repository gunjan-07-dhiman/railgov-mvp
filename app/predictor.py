"""
app/predictor.py
Machine Learning Predictor & Explainability Engine for RailGov MVP.
- Trains GradientBoostingClassifier with holdout validation.
- Extracts baseline feature distributions for drift detection.
- Generates predictions with confidence tiers.
- Produces human-readable TreeSHAP explanations without percentage claims (log-odds).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from app.config import (
    CATEGORICAL_FEATURES,
    CONFIDENCE_HIGH_MARGIN,
    CONFIDENCE_MED_MARGIN,
    FEATURE_COLUMNS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROB_CONFIRMED_THRESHOLD,
    PROB_UNCERTAIN_LOWER,
    SHAP_MODERATE_THRESHOLD,
    SHAP_STRONG_THRESHOLD,
    TARGET_COLUMN,
)


def _build_preprocessor() -> ColumnTransformer:
    """Build column transformer for categoricals (One-Hot) and numerics (passthrough)."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def extract_feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract reference statistics (quantiles, bins, and frequency counts)
    for subsequent drift detection (PSI and KS tests).
    """
    stats: Dict[str, Any] = {"numeric": {}, "categorical": {}, "sample_count": len(df)}

    for col in NUMERIC_FEATURES:
        series = df[col].dropna()
        # Compute 10-bin deciles for numeric PSI
        quantiles = np.linspace(0.0, 1.0, 11)
        bin_edges = np.unique(np.quantile(series, quantiles))
        stats["numeric"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "bin_edges": bin_edges.tolist(),
        }

    for col in CATEGORICAL_FEATURES:
        freqs = df[col].value_counts(normalize=True).to_dict()
        stats["categorical"][col] = {k: float(v) for k, v in freqs.items()}

    return stats


def train(df: pd.DataFrame, version: str) -> Dict[str, Any]:
    """
    Train a GradientBoostingClassifier on the provided dataset.

    Args:
        df: Input DataFrame containing features and target column.
        version: Version string (e.g. 'v1', 'v2').

    Returns:
        Dictionary containing holdout metrics, model paths, and feature stats.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).values

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = _build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Get transformed column names for interpretability
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    transformed_feature_names = list(cat_feature_names) + NUMERIC_FEATURES

    # Train GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train_trans, y_train)

    # Holdout validation
    y_pred_prob = model.predict_proba(X_test_trans)[:, 1]
    y_pred = (y_pred_prob >= 0.50).astype(int)

    metrics = {
        "auc": round(float(roc_auc_score(y_test, y_pred_prob)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
    }

    # Reference statistics for drift detection
    feature_stats = extract_feature_stats(X_train)

    # Save model bundle and metadata in models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{version}.joblib"
    meta_path = MODELS_DIR / f"{version}_meta.joblib"

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "transformed_feature_names": transformed_feature_names,
        "version": version,
    }
    joblib.dump(bundle, model_path)

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_stats": feature_stats,
    }
    joblib.dump(metadata, meta_path)

    return {
        "version": version,
        "metrics": metrics,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "feature_stats": feature_stats,
    }


def load_model(version: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model bundle and metadata from disk."""
    model_path = MODELS_DIR / f"{version}.joblib"
    meta_path = MODELS_DIR / f"{version}_meta.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    bundle = joblib.load(model_path)
    meta = joblib.load(meta_path) if meta_path.exists() else {}
    return bundle, meta


def _format_explanation(
    feature: str, val: Any, shap_val: float, direction: str, strength: str
) -> str:
    """Format a human-readable explanation with strength label and no percentage claims."""
    if feature == "wl_position":
        if shap_val < 0 and val > 40:
            return f"High WL position ({val}) {strength} reduced the chance of confirmation."
        elif shap_val > 0 and val <= 40:
            return f"Favourable queue position ({val}) {strength} increased the chance of confirmation."
        return f"Waitlist position of {val} {strength} {direction}."

    if feature == "booking_status":
        if val == "RAC":
            return f"RAC booking status {strength} {direction}."
        return f"Waitlist (WL) status {strength} {direction}."

    if feature == "days_before_journey":
        if val > 30 and shap_val > 0:
            return f"Early booking ({val} days before journey) {strength} increased the chance of confirmation."
        elif val <= 10 and shap_val < 0:
            return f"Short booking window ({val} days before journey) {strength} reduced the chance of confirmation."
        return f"Booking window of {val} days {strength} {direction}."

    if feature == "is_festival_season":
        if val == 1:
            return f"Festival season rush {strength} {direction}."
        return f"Off-peak travel period {strength} {direction}."

    if feature == "route_popularity":
        return f"Route demand intensity ({val:.2f}) {strength} {direction}."

    if feature == "travel_class":
        return f"Travel class {val} cancellation pattern {strength} {direction}."

    if feature == "train_type":
        return f"Train category ({val}) {strength} {direction}."

    if feature == "month":
        return f"Travel month ({val}) {strength} {direction}."

    if feature == "day_of_week":
        return f"Journey day of week ({val}) {strength} {direction}."

    return f"Feature {feature} ({val}) {strength} {direction}."


def predict(
    record: Union[Dict[str, Any], pd.DataFrame],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate ticket confirmation prediction and SHAP explanation.

    Args:
        record: Single record dictionary or 1-row DataFrame.
        bundle: Loaded model bundle dictionary.

    Returns:
        Prediction result containing probability, status, confidence, and top factors.
    """
    if isinstance(record, dict):
        df_row = pd.DataFrame([record])
    else:
        df_row = record.copy()

    model: GradientBoostingClassifier = bundle["model"]
    preprocessor: ColumnTransformer = bundle["preprocessor"]
    transformed_feature_names: List[str] = bundle["transformed_feature_names"]

    # Transform input record
    X_input = df_row[FEATURE_COLUMNS]
    X_trans = preprocessor.transform(X_input)

    # 1. Prediction probability & status
    prob = float(model.predict_proba(X_trans)[0, 1])

    if prob >= PROB_CONFIRMED_THRESHOLD:
        status = "Likely Confirmed"
    elif prob >= PROB_UNCERTAIN_LOWER:
        status = "Uncertain"
    else:
        status = "Likely Not Confirmed"

    # 2. Confidence tier based on distance from 0.5 decision boundary
    margin = abs(prob - 0.50)
    if margin >= CONFIDENCE_HIGH_MARGIN:
        confidence = "High"
    elif margin >= CONFIDENCE_MED_MARGIN:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 3. TreeSHAP Explanation in log-odds space
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_trans)

    # GradientBoosting in shap returns 1D array of log-odds for binary classification
    if isinstance(shap_vals, list):
        # Multi-class or 2-element list format
        row_shap = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif shap_vals.ndim == 2:
        row_shap = shap_vals[0]
    elif shap_vals.ndim == 3:
        row_shap = shap_vals[0, :, 1]
    else:
        row_shap = shap_vals

    # Aggregate one-hot SHAP values back to parent original features
    feature_contributions: Dict[str, float] = {f: 0.0 for f in FEATURE_COLUMNS}

    for col_name, shap_v in zip(transformed_feature_names, row_shap):
        matched = False
        for cat_col in CATEGORICAL_FEATURES:
            if col_name.startswith(f"cat__{cat_col}_") or col_name.startswith(f"{cat_col}_"):
                feature_contributions[cat_col] += float(shap_v)
                matched = True
                break
        if not matched:
            for num_col in NUMERIC_FEATURES:
                if col_name.endswith(num_col):
                    feature_contributions[num_col] += float(shap_v)
                    break

    # Rank top 3 contributors by absolute SHAP magnitude
    ranked_features = sorted(
        feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True
    )[:3]

    top_factors: List[Dict[str, Any]] = []
    for feat, s_val in ranked_features:
        val = df_row[feat].iloc[0]
        direction = (
            "increased the chance of confirmation"
            if s_val > 0
            else "reduced the chance of confirmation"
        )
        abs_s = abs(s_val)
        if abs_s >= SHAP_STRONG_THRESHOLD:
            strength = "strongly"
        elif abs_s >= SHAP_MODERATE_THRESHOLD:
            strength = "moderately"
        else:
            strength = "slightly"

        explanation_text = _format_explanation(feat, val, s_val, direction, strength)

        top_factors.append(
            {
                "feature": feat,
                "value": val if not isinstance(val, (np.integer, np.floating)) else val.item(),
                "shap_value": round(float(s_val), 4),
                "strength": strength,
                "direction": direction,
                "explanation": explanation_text,
            }
        )

    return {
        "probability": round(prob, 4),
        "predicted_status": status,
        "confidence": confidence,
        "top_factors": top_factors,
    }

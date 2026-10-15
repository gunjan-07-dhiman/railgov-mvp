"""
app/config.py
Centralized configuration for RailGov MVP.
All project constants, thresholds, paths, and guardrail limits are defined here.
"""

from pathlib import Path
from typing import List

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REGISTRY_PATH: Path = MODELS_DIR / "registry.json"
AUDIT_DB_PATH: Path = MODELS_DIR / "audit.db"

# Data schema and feature sets
TARGET_COLUMN: str = "confirmed"

FEATURE_COLUMNS: List[str] = [
    "travel_class",
    "booking_status",
    "wl_position",
    "days_before_journey",
    "day_of_week",
    "month",
    "is_festival_season",
    "route_popularity",
    "train_type",
]

CATEGORICAL_FEATURES: List[str] = ["travel_class", "booking_status", "train_type"]
NUMERIC_FEATURES: List[str] = [
    "wl_position",
    "days_before_journey",
    "day_of_week",
    "month",
    "is_festival_season",
    "route_popularity",
]

# Domain validity bounds for Data Quality checks
VALID_CLASSES: List[str] = ["SL", "3A", "2A", "1A", "CC"]
VALID_STATUSES: List[str] = ["WL", "RAC"]
VALID_TRAIN_TYPES: List[str] = ["Express", "Superfast", "Rajdhani"]

MIN_WL_POSITION: int = 1
MAX_WL_POSITION: int = 200
MAX_RAC_POSITION: int = 60
MIN_DAYS_BEFORE: int = 0
MAX_DAYS_BEFORE: int = 120
MIN_MONTH: int = 1
MAX_MONTH: int = 12
MIN_DAY_OF_WEEK: int = 0
MAX_DAY_OF_WEEK: int = 6
MIN_ROUTE_POP: float = 0.0
MAX_ROUTE_POP: float = 1.0

# Drift detection thresholds
PSI_WARNING_THRESHOLD: float = 0.10
PSI_CRITICAL_THRESHOLD: float = 0.20
KS_PVALUE_ALPHA: float = 0.05
PSI_NUM_BINS: int = 10

# Performance and retraining thresholds
AUC_DEGRADATION_THRESHOLD: float = 0.05  # Flag if AUC drops by more than 0.05
RETRAIN_PROMOTION_TOLERANCE: float = 0.005  # Promoted if candidate AUC >= active - 0.005

# Predictor thresholds
PROB_CONFIRMED_THRESHOLD: float = 0.65
PROB_UNCERTAIN_LOWER: float = 0.40
PROB_UNCERTAIN_UPPER: float = 0.65

# Confidence classification bounds (|prob - 0.5|)
CONFIDENCE_HIGH_MARGIN: float = 0.25
CONFIDENCE_MED_MARGIN: float = 0.12

# SHAP log-odds explanation strength tiers
SHAP_STRONG_THRESHOLD: float = 0.70
SHAP_MODERATE_THRESHOLD: float = 0.25

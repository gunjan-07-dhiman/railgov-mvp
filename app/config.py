"""
app/config.py
Centralized configuration for RailGov MVP.
All project constants, thresholds, paths, and guardrail limits are defined here.
"""

from pathlib import Path
from typing import List

# Branding and Application Identity
APP_NAME: str = "YatraGov AI"
HOME_TAGLINE: str = "PREDICTIVE TRAVEL & AI LIFECYCLE GOVERNANCE"
SIDEBAR_TAGLINE: str = "TRAVEL AI & GOVERNANCE"
FOOTER_DISCLAIMER: str = "Academic prototype. Uses synthetic demo data. Not affiliated with Indian Railways or IRCTC."

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

# Delay and Congestion Module Constants
DELAY_FEATURE_COLUMNS: List[str] = [
    "distance_km",
    "scheduled_hour",
    "day_of_week",
    "rainfall_mm",
    "prev_station_delay_min",
    "route_congestion_index",
    "train_type",
]
DELAY_CATEGORICAL_FEATURES: List[str] = ["train_type"]
DELAY_NUMERIC_FEATURES: List[str] = [
    "distance_km",
    "scheduled_hour",
    "day_of_week",
    "rainfall_mm",
    "prev_station_delay_min",
    "route_congestion_index",
]
DELAY_TARGET_COLUMN: str = "delay_minutes"
CONGESTION_TARGET_COLUMN: str = "congestion_level"
VALID_CONGESTION_LEVELS: List[str] = ["Low", "Moderate", "High"]

MIN_DISTANCE_KM: float = 10.0
MAX_DISTANCE_KM: float = 3000.0
MIN_SCHEDULED_HOUR: int = 0
MAX_SCHEDULED_HOUR: int = 23
MIN_RAINFALL_MM: float = 0.0
MAX_RAINFALL_MM: float = 200.0
MIN_PREV_DELAY_MIN: float = 0.0
MAX_PREV_DELAY_MIN: float = 300.0

MAX_DELAY_MAE_DEGRADATION: float = 5.0  # Flag if holdout MAE increases by > 5 minutes
PROACTIVE_ALERT_DELAY_THRESHOLD: int = 15  # Minutes of delay to trigger passenger warning

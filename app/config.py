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

# Fare Prediction & Demand Surge Module Constants
FARE_FEATURE_COLUMNS: List[str] = [
    "travel_class",
    "train_type",
    "distance_km",
    "days_before_journey",
    "day_of_week",
    "month",
    "is_festival_season",
    "route_popularity",
    "seats_available_pct",
]
FARE_CATEGORICAL_FEATURES: List[str] = ["travel_class", "train_type"]
FARE_NUMERIC_FEATURES: List[str] = [
    "distance_km",
    "days_before_journey",
    "day_of_week",
    "month",
    "is_festival_season",
    "route_popularity",
    "seats_available_pct",
]
FARE_TARGET_COLUMN: str = "fare_inr"
DEMAND_TARGET_COLUMN: str = "demand_level"
VALID_DEMAND_LEVELS: List[str] = ["Normal", "High", "Surge"]

MIN_SEATS_PCT: float = 0.0
MAX_SEATS_PCT: float = 100.0
MIN_FARE_INR: float = 50.0
MAX_FARE_INR: float = 20000.0

BOOKING_HORIZON_DAYS: int = 60
FARE_ANOMALY_THRESHOLD: float = 0.30  # 30% deviation triggers price anomaly flag
MAX_FARE_MAE_DEGRADATION: float = 150.0  # Flag if holdout MAE increases by > 150 INR

FARE_BASELINE_PATH: Path = DATA_DIR / "fare_baseline.csv"
FARE_DRIFTED_PATH: Path = DATA_DIR / "fare_drifted.csv"

# Centralized Passenger Module Catalog (Single Source of Truth for Nav & Cards)
MODULE_CATALOG: List[dict] = [
    {
        "id": "ticket_confirmation",
        "title": "Ticket Confirmation Predictor",
        "short_title": "Ticket Confirmation",
        "page": "pages/ticket_confirmation.py",
        "desc": "Waitlist & RAC confirmation odds with TreeSHAP explainability drivers.",
        "built": True,
    },
    {
        "id": "delay",
        "title": "Train Delay & Platform Crowd",
        "short_title": "Train Delay & Platform",
        "page": "pages/delay.py",
        "desc": "Regression delay minutes range, station crowd classifier, and proactive arrival advice.",
        "built": True,
    },
    {
        "id": "fare",
        "title": "Dynamic Fare Movement",
        "short_title": "Fare & Demand Surge",
        "page": "pages/fare.py",
        "desc": "Demand surge tracking, best purchase horizon scan, and price anomaly detection.",
        "built": True,
    },
    {
        "id": "fraud",
        "title": "Booking Fraud Shield",
        "short_title": "Fraud Shield",
        "page": None,
        "desc": "High-frequency automated bot detection and risk classification.",
        "built": False,
    },
    {
        "id": "support",
        "title": "Support Chatbot",
        "short_title": "Support Chatbot",
        "page": None,
        "desc": "Grounded multilingual travel guidance and Indian script detection.",
        "built": False,
    },
    {
        "id": "recommender",
        "title": "Train Recommender",
        "short_title": "Train Recommender",
        "page": None,
        "desc": "Fair alternate train route suggestions with audit logs.",
        "built": False,
    },
    {
        "id": "planner",
        "title": "Journey Planner Agent",
        "short_title": "Journey Planner Agent",
        "page": None,
        "desc": "Autonomous multi-leg journey coordination with user approval gates.",
        "built": False,
    },
]

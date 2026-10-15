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

# Third-Party Live Rail API Configuration (RapidAPI)
import os

def _load_env_vars():
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env_vars()
RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY") or os.getenv("X_RAPIDAPI_KEY") or ""
RAPIDAPI_HOST: str = os.getenv("RAPIDAPI_HOST") or os.getenv("X_RAPIDAPI_HOST") or "irctc1.p.rapidapi.com"

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

# Fraud Shield & Bot Detection Module Constants
FRAUD_FEATURE_COLUMNS: List[str] = [
    "bookings_last_hour",
    "accounts_per_device",
    "accounts_per_ip",
    "form_fill_seconds",
    "failed_payments",
    "account_age_days",
    "is_tatkal_window",
    "avg_gap_between_clicks_ms",
]
FRAUD_CATEGORICAL_FEATURES: List[str] = ["is_tatkal_window"]
FRAUD_NUMERIC_FEATURES: List[str] = [
    "bookings_last_hour",
    "accounts_per_device",
    "accounts_per_ip",
    "form_fill_seconds",
    "failed_payments",
    "account_age_days",
    "avg_gap_between_clicks_ms",
]
FRAUD_TARGET_COLUMN: str = "is_fraud"

# Fraud Risk Thresholds (Score 0-100)
FRAUD_RISK_LOW_THRESHOLD: float = 30.0    # Score < 30 -> Low Risk
FRAUD_RISK_HIGH_THRESHOLD: float = 70.0   # Score >= 70 -> High Risk (30-69 -> Medium Risk)
MAX_FRAUD_RECALL_DROP: float = 0.08       # Flag degradation if holdout recall drops by > 0.08 (8%)

FRAUD_BASELINE_PATH: Path = DATA_DIR / "fraud_baseline.csv"
FRAUD_DRIFTED_PATH: Path = DATA_DIR / "fraud_drifted.csv"

# Support Chatbot Module Constants
FAQ_DIR: Path = BASE_DIR / "docs" / "faq"
MOCK_PNR_PATH: Path = DATA_DIR / "mock_pnr.csv"
CHAT_EVAL_PATH: Path = DATA_DIR / "chat_eval.csv"
CHAT_BASELINE_PATH: Path = DATA_DIR / "chat_baseline.csv"
CHAT_DRIFTED_PATH: Path = DATA_DIR / "chat_drifted.csv"

# Grounding threshold determined via tuning split calibration
CHAT_SIMILARITY_THRESHOLD: float = 0.20   # Queries with cosine sim < 0.20 trigger "not sure" response & escalation
MIN_CHAT_GROUNDED_RATE: float = 0.65      # Alert if grounded response rate drops below 65%

CHAT_FEATURE_COLUMNS: List[str] = ["language", "char_length", "has_pnr", "similarity_score", "is_grounded"]
CHAT_CATEGORICAL_FEATURES: List[str] = ["language", "has_pnr"]
CHAT_NUMERIC_FEATURES: List[str] = ["char_length", "similarity_score", "is_grounded"]
CHAT_TARGET_COLUMN: str = "is_grounded"

# Centralized Passenger Module Catalog (Single Source of Truth for Nav & Cards)
# Exactly matching the 6 Core AI Features - All 6 Built & Active
MODULE_CATALOG: List[dict] = [
    {
        "id": "ticket_confirmation",
        "num": 1,
        "title": "Ticket Confirm Predictor",
        "short_title": "Ticket Confirm Predictor",
        "page": "pages/ticket_confirmation.py",
        "desc": "Tells the passenger: 'Your WL 12 has an 82% chance of confirming'.",
        "built": True,
    },
    {
        "id": "recommender",
        "num": 2,
        "title": "Train Recommender",
        "short_title": "Train Recommender",
        "page": "pages/recommender.py",
        "desc": "Suggests the best train based on what you care about (fastest, cheapest, cleanest).",
        "built": True,
    },
    {
        "id": "delay",
        "num": 3,
        "title": "Delay Predictor",
        "short_title": "Delay Predictor",
        "page": "pages/delay.py",
        "desc": "Tells the passenger: 'This train usually gets delayed by 2 hours at Kanpur due to winter fog'.",
        "built": True,
    },
    {
        "id": "fare",
        "num": 4,
        "title": "Dynamic Fare Predictor",
        "short_title": "Dynamic Fare Predictor",
        "page": "pages/fare.py",
        "desc": "Predicts when ticket prices or Tatkal demand will surge.",
        "built": True,
    },
    {
        "id": "support",
        "num": 5,
        "title": "24/7 AI Travel Assistant & Planner",
        "short_title": "24/7 AI Assistant & Planner",
        "page": "pages/support_chat.py",
        "desc": "Ask travel plans ('Pune to Delhi under ₹1500'), live PNR status, and refund rules in English & हिंदी.",
        "built": True,
    },
    {
        "id": "fraud",
        "num": 6,
        "title": "Anti-Cheat / Anti-Bot System",
        "short_title": "Anti-Cheat / Anti-Bot",
        "page": "pages/fraud.py",
        "desc": "Catches scalpers and automated bots booking all the Tatkal tickets in 5 seconds.",
        "built": True,
    },
]

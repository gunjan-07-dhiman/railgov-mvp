"""
app/governance/data_quality.py
Data Quality Guardrail for RailGov MVP.
Pre-inference input validation to block anomalous, malformed, or out-of-domain records.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

from app.config import (
    FEATURE_COLUMNS,
    MAX_DAYS_BEFORE,
    MAX_DAY_OF_WEEK,
    MAX_MONTH,
    MAX_RAC_POSITION,
    MAX_ROUTE_POP,
    MAX_WL_POSITION,
    MIN_DAYS_BEFORE,
    MIN_DAY_OF_WEEK,
    MIN_MONTH,
    MIN_ROUTE_POP,
    MIN_WL_POSITION,
    VALID_CLASSES,
    VALID_STATUSES,
    VALID_TRAIN_TYPES,
)


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single ticket booking record before running model inference.

    Args:
        record: Dictionary containing ticket feature key-value pairs.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []

    # 1. Missing field verification
    for col in FEATURE_COLUMNS:
        if col not in record or record[col] is None:
            issues.append(f"Missing required field: '{col}'")

    if issues:
        return False, issues

    # 2. Categorical domain checks
    travel_class = record.get("travel_class")
    if travel_class not in VALID_CLASSES:
        issues.append(
            f"Invalid travel_class: '{travel_class}'. Must be one of {VALID_CLASSES}"
        )

    booking_status = record.get("booking_status")
    if booking_status not in VALID_STATUSES:
        issues.append(
            f"Invalid booking_status: '{booking_status}'. Must be one of {VALID_STATUSES}"
        )

    train_type = record.get("train_type")
    if train_type not in VALID_TRAIN_TYPES:
        issues.append(
            f"Invalid train_type: '{train_type}'. Must be one of {VALID_TRAIN_TYPES}"
        )

    # 3. Numeric range and type checks
    try:
        wl_pos = int(record["wl_position"])
        if booking_status == "RAC":
            if not (MIN_WL_POSITION <= wl_pos <= MAX_RAC_POSITION):
                issues.append(
                    f"RAC wl_position must be between {MIN_WL_POSITION} and {MAX_RAC_POSITION}, got {wl_pos}"
                )
        else:
            if not (MIN_WL_POSITION <= wl_pos <= MAX_WL_POSITION):
                issues.append(
                    f"wl_position must be between {MIN_WL_POSITION} and {MAX_WL_POSITION}, got {wl_pos}"
                )
    except (ValueError, TypeError):
        issues.append(f"wl_position must be an integer, got '{record.get('wl_position')}'")

    try:
        days_before = int(record["days_before_journey"])
        if not (MIN_DAYS_BEFORE <= days_before <= MAX_DAYS_BEFORE):
            issues.append(
                f"days_before_journey must be between {MIN_DAYS_BEFORE} and {MAX_DAYS_BEFORE}, got {days_before}"
            )
    except (ValueError, TypeError):
        issues.append(
            f"days_before_journey must be an integer, got '{record.get('days_before_journey')}'"
        )

    try:
        dow = int(record["day_of_week"])
        if not (MIN_DAY_OF_WEEK <= dow <= MAX_DAY_OF_WEEK):
            issues.append(
                f"day_of_week must be between {MIN_DAY_OF_WEEK} and {MAX_DAY_OF_WEEK}, got {dow}"
            )
    except (ValueError, TypeError):
        issues.append(f"day_of_week must be an integer, got '{record.get('day_of_week')}'")

    try:
        month = int(record["month"])
        if not (MIN_MONTH <= month <= MAX_MONTH):
            issues.append(f"month must be between {MIN_MONTH} and {MAX_MONTH}, got {month}")
    except (ValueError, TypeError):
        issues.append(f"month must be an integer, got '{record.get('month')}'")

    try:
        fest = int(record["is_festival_season"])
        if fest not in (0, 1):
            issues.append(f"is_festival_season must be 0 or 1, got {fest}")
    except (ValueError, TypeError):
        issues.append(
            f"is_festival_season must be 0 or 1, got '{record.get('is_festival_season')}'"
        )

    try:
        pop = float(record["route_popularity"])
        if not (MIN_ROUTE_POP <= pop <= MAX_ROUTE_POP):
            issues.append(
                f"route_popularity must be between {MIN_ROUTE_POP} and {MAX_ROUTE_POP}, got {pop}"
            )
    except (ValueError, TypeError):
        issues.append(
            f"route_popularity must be a float, got '{record.get('route_popularity')}'"
        )

    return len(issues) == 0, issues


def validate_batch(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate a batch DataFrame of ticket records.

    Args:
        df: Input DataFrame to check.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []

    # Check required columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Batch is missing required columns: {missing_cols}")
        return False, issues

    # Null checks
    null_counts = df[FEATURE_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Batch contains null values: {cols_with_nulls.to_dict()}")

    # Categorical domain checks
    invalid_classes = set(df["travel_class"].dropna().unique()) - set(VALID_CLASSES)
    if invalid_classes:
        issues.append(f"Found invalid travel_classes in batch: {list(invalid_classes)}")

    invalid_statuses = set(df["booking_status"].dropna().unique()) - set(VALID_STATUSES)
    if invalid_statuses:
        issues.append(f"Found invalid booking_statuses in batch: {list(invalid_statuses)}")

    invalid_train_types = set(df["train_type"].dropna().unique()) - set(VALID_TRAIN_TYPES)
    if invalid_train_types:
        issues.append(f"Found invalid train_types in batch: {list(invalid_train_types)}")

    # Numeric range checks
    bad_wl = df[(df["wl_position"] < MIN_WL_POSITION) | (df["wl_position"] > MAX_WL_POSITION)]
    if len(bad_wl) > 0:
        issues.append(f"{len(bad_wl)} records with wl_position out of range [1, 200]")

    bad_days = df[
        (df["days_before_journey"] < MIN_DAYS_BEFORE)
        | (df["days_before_journey"] > MAX_DAYS_BEFORE)
    ]
    if len(bad_days) > 0:
        issues.append(f"{len(bad_days)} records with days_before_journey out of range [0, 120]")

    return len(issues) == 0, issues

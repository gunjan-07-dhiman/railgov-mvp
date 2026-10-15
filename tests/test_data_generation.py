"""
tests/test_data_generation.py
Tests for synthetic data generation:
- Schema verification (required columns and non-null values)
- Baseline vs. Drifted statistical properties (festival season surge, WL distribution)
- Valid label generation (binary confirmed flag with healthy balance)
- Invalid samples coverage for data quality testing
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from data.generate_data import (
    CLASSES,
    STATUSES,
    TRAIN_TYPES,
    generate_bookings,
    generate_invalid_samples,
)


def test_baseline_generation_schema():
    df = generate_bookings(n=500, drifted=False, seed=42)
    expected_cols = [
        "booking_id",
        "travel_class",
        "booking_status",
        "wl_position",
        "days_before_journey",
        "day_of_week",
        "month",
        "is_festival_season",
        "route_popularity",
        "train_type",
        "confirmed",
    ]
    assert len(df) == 500
    assert list(df.columns) == expected_cols
    assert df.isnull().sum().sum() == 0

    # Categorical domain checks
    assert set(df["travel_class"].unique()).issubset(set(CLASSES))
    assert set(df["booking_status"].unique()).issubset(set(STATUSES))
    assert set(df["train_type"].unique()).issubset(set(TRAIN_TYPES))

    # Numeric range checks
    assert df["wl_position"].min() >= 1
    assert df["wl_position"].max() <= 200
    assert df["days_before_journey"].min() >= 0
    assert df["days_before_journey"].max() <= 120
    assert df["day_of_week"].min() >= 0
    assert df["day_of_week"].max() <= 6
    assert df["month"].min() >= 1
    assert df["month"].max() <= 12
    assert set(df["is_festival_season"].unique()).issubset({0, 1})
    assert set(df["confirmed"].unique()).issubset({0, 1})
    assert 0.0 <= df["route_popularity"].min() <= 1.0
    assert 0.0 <= df["route_popularity"].max() <= 1.0


def test_drifted_generation_distribution_shift():
    baseline_df = generate_bookings(n=2000, drifted=False, seed=42)
    drifted_df = generate_bookings(n=2000, drifted=True, seed=99)

    # 1. Festival season ratio surges from ~18% to ~70%
    assert baseline_df["is_festival_season"].mean() < 0.30
    assert drifted_df["is_festival_season"].mean() > 0.60

    # 2. Average WL queue is significantly higher in drifted data
    assert drifted_df["wl_position"].mean() > baseline_df["wl_position"].mean()

    # 3. Route popularity is noticeably higher in drifted data
    assert drifted_df["route_popularity"].mean() > baseline_df["route_popularity"].mean()

    # 4. Confirmation rate drops significantly due to festival lock-in
    assert drifted_df["confirmed"].mean() < baseline_df["confirmed"].mean()


def test_seed_reproducibility():
    df1 = generate_bookings(n=100, drifted=False, seed=123)
    df2 = generate_bookings(n=100, drifted=False, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_invalid_samples_structure():
    samples = generate_invalid_samples()
    assert len(samples) >= 5
    for s in samples:
        assert "test_case" in s
        assert "record" in s
        assert "expected_issue" in s
        assert isinstance(s["record"], dict)

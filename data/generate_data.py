"""
data/generate_data.py
Generates synthetic Indian Railways ticket booking datasets for RailGov MVP:
- data/baseline.csv (5000 rows): Normal baseline operations.
- data/drifted.csv (3000 rows): Shifted operational conditions (festival rush, lower cancellations).
- data/invalid_samples.json: Malformed records for data-quality guardrail validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

CLASSES = ["SL", "3A", "2A", "1A", "CC"]
STATUSES = ["WL", "RAC"]
TRAIN_TYPES = ["Express", "Superfast", "Rajdhani"]


def generate_bookings(n: int = 5000, drifted: bool = False, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ticket booking records with realistic Indian Railways dynamics.

    Args:
        n: Number of records to generate.
        drifted: If True, simulates festival season surge with high route popularity,
                 crowded WL queues, last-minute bookings, and reduced cancellations.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing ticket features and confirmation label.
    """
    rng = np.random.default_rng(seed)

    # 1. Booking IDs
    booking_ids = np.arange(100000, 100000 + n)

    # 2. Travel class & Booking status
    if not drifted:
        class_probs = [0.42, 0.30, 0.15, 0.05, 0.08]
        status_probs = [0.70, 0.30]  # 70% WL, 30% RAC
    else:
        # Drifted: Higher proportion of sleeper and 3A under heavy rush
        class_probs = [0.48, 0.32, 0.12, 0.03, 0.05]
        status_probs = [0.85, 0.15]  # RAC seats quickly exhausted, WL predominates

    travel_classes = rng.choice(CLASSES, size=n, p=class_probs)
    booking_statuses = rng.choice(STATUSES, size=n, p=status_probs)

    # 3. Queue positions: WL 1-200, RAC 1-60
    wl_positions = np.zeros(n, dtype=int)
    for i in range(n):
        if booking_statuses[i] == "RAC":
            if not drifted:
                wl_positions[i] = int(rng.integers(1, 61))
            else:
                # Drifted: RAC positions skewed towards the tail (30-60)
                wl_positions[i] = int(rng.integers(20, 61))
        else:
            if not drifted:
                # Baseline: exponential distribution skewed towards lower WL
                pos = int(rng.exponential(scale=35.0) + 1)
                wl_positions[i] = min(max(pos, 1), 200)
            else:
                # Drifted: Heavily backed-up waitlists (mean shifted higher)
                pos = int(rng.normal(loc=95.0, scale=35.0))
                wl_positions[i] = min(max(pos, 1), 200)

    # 4. Journey horizons: days_before_journey (0-120)
    if not drifted:
        # Normal booking window spread
        days_before_journey = rng.integers(0, 121, size=n)
    else:
        # Drifted: Rush bookings happen closer to journey date
        days_before_journey = np.clip(
            rng.exponential(scale=20.0, size=n).astype(int), 0, 120
        )

    # 5. Temporal features
    if not drifted:
        months = rng.integers(1, 13, size=n)
        days_of_week = rng.integers(0, 7, size=n)
        # Baseline festival season: ~18% (e.g. Oct/Nov or holiday periods)
        is_festival_season = rng.choice([0, 1], size=n, p=[0.82, 0.18])
        # Baseline route popularity: centered around 0.50
        route_popularity = np.clip(rng.beta(a=2.0, b=2.0, size=n), 0.05, 0.95)
    else:
        # Drifted: Concentrated in festival period (months 10-11, ~72% festival season)
        months = rng.choice([10, 11, 12], size=n, p=[0.50, 0.40, 0.10])
        days_of_week = rng.integers(0, 7, size=n)
        is_festival_season = rng.choice([0, 1], size=n, p=[0.28, 0.72])
        # Drifted route popularity: skewed high (~0.75 mean)
        route_popularity = np.clip(rng.beta(a=5.0, b=1.8, size=n), 0.15, 0.99)

    # 6. Train types
    train_types = rng.choice(TRAIN_TYPES, size=n, p=[0.50, 0.35, 0.15])

    # 7. Confirmation Probability Model (Learnable Ground Truth)
    # Class effects on cancellation volume (SL/3A have higher cancellation turnover)
    class_multipliers = {"SL": 0.60, "3A": 0.35, "CC": 0.00, "2A": -0.35, "1A": -0.75}
    train_type_multipliers = {"Express": 0.20, "Superfast": 0.00, "Rajdhani": -0.40}

    class_contrib = np.array([class_multipliers[c] for c in travel_classes])
    train_contrib = np.array([train_type_multipliers[t] for t in train_types])

    if not drifted:
        # Baseline probability logit
        status_boost = np.where(booking_statuses == "RAC", 2.2, 0.0)
        wl_penalty = np.where(
            booking_statuses == "RAC",
            -0.030 * wl_positions,
            -0.038 * wl_positions,
        )
        days_benefit = 0.025 * days_before_journey
        festival_penalty = -1.10 * is_festival_season
        route_penalty = -1.40 * route_popularity
        noise = rng.normal(loc=0.0, scale=0.75, size=n)

        logit = (
            0.90
            + status_boost
            + wl_penalty
            + days_benefit
            + festival_penalty
            + route_penalty
            + class_contrib
            + train_contrib
            + noise
        )
    else:
        # Drifted: Concept drift under peak festival rush.
        # 1. RAC boost plummets: confirmed passengers do not cancel, so RAC berths rarely clear.
        status_boost = np.where(booking_statuses == "RAC", 0.20, 0.0)
        # 2. Inverted class cancellation dynamics: Sleeper and 3A cancellations freeze,
        # while corporate/VIP cancellations occur more in 1A and 2A.
        drifted_class_multipliers = {"SL": -0.70, "3A": -0.30, "CC": 0.00, "2A": 0.50, "1A": 0.85}
        class_contrib = np.array([drifted_class_multipliers[c] for c in travel_classes])
        # 3. Booking window decoupling: booking early provides little advantage when queues are locked
        days_benefit = 0.002 * days_before_journey + 0.40 * (days_before_journey < 5)
        wl_penalty = -0.028 * wl_positions
        festival_penalty = -1.30 * is_festival_season
        route_penalty = -1.20 * route_popularity
        noise = rng.normal(loc=0.0, scale=1.35, size=n)

        logit = (
            0.20
            + status_boost
            + wl_penalty
            + days_benefit
            + festival_penalty
            + route_penalty
            + class_contrib
            + train_contrib
            + noise
        )

    # Sigmoid function
    prob = 1.0 / (1.0 + np.exp(-logit))
    # Binary confirmation label
    confirmed = (rng.uniform(0.0, 1.0, size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "booking_id": booking_ids,
            "travel_class": travel_classes,
            "booking_status": booking_statuses,
            "wl_position": wl_positions,
            "days_before_journey": days_before_journey,
            "day_of_week": days_of_week,
            "month": months,
            "is_festival_season": is_festival_season,
            "route_popularity": np.round(route_popularity, 4),
            "train_type": train_types,
            "confirmed": confirmed,
        }
    )
    return df


def generate_invalid_samples() -> List[Dict]:
    """
    Generate malformed sample records to test Data Quality guardrails:
    - Missing required fields
    - Invalid categorical values
    - Negative numeric values (wl_position: -5)
    - Unrealistic journey horizons (days_before_journey: 900)
    """
    return [
        {
            "test_case": "missing_travel_class",
            "record": {
                "booking_id": 999001,
                "booking_status": "WL",
                "wl_position": 45,
                "days_before_journey": 20,
                "day_of_week": 2,
                "month": 6,
                "is_festival_season": 0,
                "route_popularity": 0.55,
                "train_type": "Superfast",
            },
            "expected_issue": "Missing field: travel_class",
        },
        {
            "test_case": "invalid_travel_class",
            "record": {
                "booking_id": 999002,
                "travel_class": "XX",
                "booking_status": "WL",
                "wl_position": 25,
                "days_before_journey": 15,
                "day_of_week": 4,
                "month": 7,
                "is_festival_season": 0,
                "route_popularity": 0.40,
                "train_type": "Express",
            },
            "expected_issue": "Invalid travel_class: XX",
        },
        {
            "test_case": "negative_wl_position",
            "record": {
                "booking_id": 999003,
                "travel_class": "3A",
                "booking_status": "WL",
                "wl_position": -5,
                "days_before_journey": 30,
                "day_of_week": 1,
                "month": 8,
                "is_festival_season": 0,
                "route_popularity": 0.65,
                "train_type": "Rajdhani",
            },
            "expected_issue": "wl_position must be between 1 and 200",
        },
        {
            "test_case": "out_of_range_days_before_journey",
            "record": {
                "booking_id": 999004,
                "travel_class": "SL",
                "booking_status": "WL",
                "wl_position": 50,
                "days_before_journey": 900,
                "day_of_week": 5,
                "month": 9,
                "is_festival_season": 0,
                "route_popularity": 0.70,
                "train_type": "Express",
            },
            "expected_issue": "days_before_journey must be between 0 and 120",
        },
        {
            "test_case": "invalid_booking_status",
            "record": {
                "booking_id": 999005,
                "travel_class": "2A",
                "booking_status": "CNF",
                "wl_position": 10,
                "days_before_journey": 10,
                "day_of_week": 0,
                "month": 5,
                "is_festival_season": 0,
                "route_popularity": 0.50,
                "train_type": "Superfast",
            },
            "expected_issue": "Invalid booking_status: CNF (must be WL or RAC)",
        },
    ]


def main() -> None:
    data_dir = Path(__file__).resolve().parent

    print("Generating baseline dataset (5000 rows)...")
    baseline_df = generate_bookings(n=5000, drifted=False, seed=42)
    baseline_path = data_dir / "baseline.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f" Saved baseline dataset to {baseline_path}")
    print(f"  Confirmation rate: {baseline_df['confirmed'].mean():.2%}")
    print(f"  Festival ratio: {baseline_df['is_festival_season'].mean():.2%}")
    print(f"  Average WL position: {baseline_df['wl_position'].mean():.1f}")

    print("\nGenerating drifted dataset (3000 rows)...")
    drifted_df = generate_bookings(n=3000, drifted=True, seed=99)
    drifted_path = data_dir / "drifted.csv"
    drifted_df.to_csv(drifted_path, index=False)
    print(f" Saved drifted dataset to {drifted_path}")
    print(f"  Confirmation rate: {drifted_df['confirmed'].mean():.2%}")
    print(f"  Festival ratio: {drifted_df['is_festival_season'].mean():.2%}")
    print(f"  Average WL position: {drifted_df['wl_position'].mean():.1f}")

    print("\nGenerating invalid samples...")
    invalid_samples = generate_invalid_samples()
    invalid_path = data_dir / "invalid_samples.json"
    with open(invalid_path, "w", encoding="utf-8") as f:
        json.dump(invalid_samples, f, indent=2)
    print(f" Saved {len(invalid_samples)} invalid samples to {invalid_path}")


if __name__ == "__main__":
    main()

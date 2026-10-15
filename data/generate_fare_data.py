"""
data/generate_fare_data.py
Synthetic data generator for Dynamic Fare Movement & Demand Surge module.
Generates baseline (normal travel conditions) and drifted (peak festive season & surge pricing) datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from app.config import (
    DATA_DIR,
    DEMAND_TARGET_COLUMN,
    FARE_BASELINE_PATH,
    FARE_DRIFTED_PATH,
    FARE_FEATURE_COLUMNS,
    FARE_TARGET_COLUMN,
    VALID_CLASSES,
    VALID_DEMAND_LEVELS,
    VALID_TRAIN_TYPES,
)

# Base tariff rate per km by travel class
BASE_RATE_PER_KM = {
    "SL": 0.45,
    "CC": 1.10,
    "3A": 1.45,
    "2A": 2.10,
    "1A": 3.40,
}

TRAIN_TYPE_SURCHARGE = {
    "Express": 0.0,
    "Superfast": 45.0,
    "Rajdhani": 180.0,
}


def generate_fare_dataset(
    n_samples: int = 2000,
    drifted: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic railway fare and demand level dataset."""
    rng = np.random.RandomState(seed)

    # 1. Travel Class & Train Type
    classes = rng.choice(VALID_CLASSES, size=n_samples, p=[0.35, 0.25, 0.20, 0.05, 0.15])
    train_types = rng.choice(VALID_TRAIN_TYPES, size=n_samples, p=[0.45, 0.40, 0.15])
    distance_km = rng.uniform(80.0, 1800.0, n_samples).round(1)

    if drifted:
        # Covariate shift: festival-heavy demand, higher popularity, fewer seats, shorter lead time
        is_festival_season = rng.choice([0, 1], size=n_samples, p=[0.15, 0.85])
        route_popularity = rng.beta(a=5.0, b=1.5, size=n_samples).round(2)
        seats_available_pct = rng.beta(a=1.5, b=5.0, size=n_samples) * 100.0  # Heavily skewed towards low seats
        days_before_journey = rng.exponential(scale=10.0, size=n_samples).clip(0, 120).astype(int)
        month = rng.choice([10, 11, 12], size=n_samples)  # Diwali / Winter festival peak
        day_of_week = rng.choice(list(range(7)), size=n_samples, p=[0.10, 0.10, 0.10, 0.12, 0.18, 0.22, 0.18])
    else:
        # Normal baseline operations
        is_festival_season = rng.choice([0, 1], size=n_samples, p=[0.80, 0.20])
        route_popularity = rng.beta(a=2.0, b=2.5, size=n_samples).round(2)
        seats_available_pct = rng.uniform(10.0, 95.0, n_samples)
        days_before_journey = rng.randint(0, 121, n_samples)
        month = rng.randint(1, 13, n_samples)
        day_of_week = rng.randint(0, 7, n_samples)

    seats_available_pct = np.clip(seats_available_pct.round(1), 0.0, 100.0)

    # 2. Base Fare Calculation
    rate_arr = np.array([BASE_RATE_PER_KM[c] for c in classes])
    surch_arr = np.array([TRAIN_TYPE_SURCHARGE[t] for t in train_types])
    base_fare = 80.0 + (distance_km * rate_arr) + surch_arr

    # 3. Dynamic Multipliers (urgency, seat scarcity, popularity, festivals)
    urgency_mult = 1.0 + np.where(days_before_journey <= 3, 0.35, np.where(days_before_journey <= 10, 0.20, np.where(days_before_journey <= 30, 0.08, 0.0)))
    scarcity_mult = 1.0 + (1.0 - (seats_available_pct / 100.0)) * 0.35
    festival_mult = 1.0 + (is_festival_season * 0.22)
    popularity_mult = 1.0 + (route_popularity * 0.15)
    weekend_mult = np.where(np.isin(day_of_week, [5, 6]), 1.08, 1.0)

    dynamic_mult = urgency_mult * scarcity_mult * festival_mult * popularity_mult * weekend_mult

    # Structural price shift on drifted data (simulating dynamic surge surcharge)
    drift_addon = 250.0 if drifted else 0.0

    raw_fare = (base_fare * dynamic_mult) + drift_addon + rng.normal(0.0, 35.0, n_samples)
    fare_inr = np.clip(np.round(raw_fare, -1), 60.0, 18000.0)

    # 4. Synthesize Demand Level (Normal / High / Surge)
    demand_score = (
        0.35 * (1.0 - (seats_available_pct / 100.0))
        + 0.25 * is_festival_season
        + 0.20 * route_popularity
        + 0.20 * np.where(days_before_journey <= 7, 1.0, 0.2)
        + rng.normal(0.0, 0.06, n_samples)
    )

    demand_levels = []
    for sc in demand_score:
        if sc >= 0.58:
            demand_levels.append("Surge")
        elif sc >= 0.38:
            demand_levels.append("High")
        else:
            demand_levels.append("Normal")

    # Inject 4% noise for realistic stochasticity
    noise_idx = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)
    for idx in noise_idx:
        demand_levels[idx] = rng.choice(VALID_DEMAND_LEVELS)

    df = pd.DataFrame({
        "travel_class": classes,
        "train_type": train_types,
        "distance_km": distance_km,
        "days_before_journey": days_before_journey,
        "day_of_week": day_of_week,
        "month": month,
        "is_festival_season": is_festival_season,
        "route_popularity": route_popularity,
        "seats_available_pct": seats_available_pct,
        FARE_TARGET_COLUMN: fare_inr,
        DEMAND_TARGET_COLUMN: demand_levels,
    })

    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_df = generate_fare_dataset(n_samples=2000, drifted=False, seed=42)
    drift_df = generate_fare_dataset(n_samples=2000, drifted=True, seed=42)

    base_df.to_csv(FARE_BASELINE_PATH, index=False)
    drift_df.to_csv(FARE_DRIFTED_PATH, index=False)
    print(f"Generated baseline fare data ({len(base_df)} rows) -> {FARE_BASELINE_PATH}")
    print(f"Generated drifted fare data ({len(drift_df)} rows) -> {FARE_DRIFTED_PATH}")

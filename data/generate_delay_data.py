"""
data/generate_delay_data.py
Synthetic data generator for Train Delay & Station Congestion module.
Generates baseline (normal operations) and drifted (monsoon/severe fog) datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from app.config import (
    DATA_DIR,
    DELAY_FEATURE_COLUMNS,
    DELAY_TARGET_COLUMN,
    CONGESTION_TARGET_COLUMN,
    VALID_CONGESTION_LEVELS,
    VALID_TRAIN_TYPES,
)


def generate_delay_dataset(
    n_samples: int = 2000,
    drifted: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic train delay and station congestion records."""
    rng = np.random.RandomState(seed)

    # 1. Generate Covariates
    distance_km = rng.uniform(80.0, 1600.0, n_samples).round(1)
    day_of_week = rng.randint(0, 7, n_samples)
    train_types = rng.choice(VALID_TRAIN_TYPES, size=n_samples, p=[0.45, 0.40, 0.15])

    if drifted:
        # Severe monsoon & fog covariate shift: heavier rain, major upstream delays, peak congestion
        probs = np.array([0.02] * 6 + [0.06] * 6 + [0.04] * 6 + [0.046] * 6)
        probs = probs / probs.sum()
        scheduled_hour = rng.choice(list(range(24)), size=n_samples, p=probs)
        rainfall_mm = rng.exponential(scale=45.0, size=n_samples).clip(0.0, 180.0).round(1)
        prev_delay_min = rng.exponential(scale=35.0, size=n_samples).clip(0.0, 240.0).round(1)
        route_congestion = rng.beta(a=4.0, b=2.0, size=n_samples).round(2)
    else:
        # Normal baseline operations
        scheduled_hour = rng.randint(0, 24, n_samples)
        rainfall_mm = rng.exponential(scale=5.0, size=n_samples).clip(0.0, 40.0).round(1)
        prev_delay_min = rng.exponential(scale=8.0, size=n_samples).clip(0.0, 90.0).round(1)
        route_congestion = rng.beta(a=2.0, b=3.0, size=n_samples).round(2)

    # 2. Synthesize Delay Minutes (Regression Ground Truth)
    type_effect = np.where(train_types == "Rajdhani", -6.0, np.where(train_types == "Superfast", 0.0, 8.0))
    peak_hour_effect = np.where(np.isin(scheduled_hour, [8, 9, 10, 17, 18, 19, 20]), 12.0, 0.0)

    delay_raw = (
        0.015 * distance_km
        + 1.1 * rainfall_mm
        + 0.75 * prev_delay_min
        + 25.0 * route_congestion
        + type_effect
        + peak_hour_effect
        + rng.normal(0, 5.0, n_samples)
    )
    delay_minutes = np.maximum(0.0, delay_raw).round(1)

    # 3. Synthesize Station Congestion Label with realistic noise
    # Higher route congestion, upstream delay, and weather drive station platform congestion
    congestion_score = (
        0.50 * route_congestion
        + 0.35 * (prev_delay_min / 90.0)
        + 0.15 * (rainfall_mm / 40.0)
        + rng.normal(0, 0.04, n_samples)
    )
    congestion_levels = []
    for score in congestion_score:
        if score >= 0.55:
            congestion_levels.append("High")
        elif score >= 0.32:
            congestion_levels.append("Moderate")
        else:
            congestion_levels.append("Low")

    # Inject 4% random noise to reflect occasional station anomalies
    noise_mask = rng.rand(n_samples) < 0.04
    for i in np.where(noise_mask)[0]:
        congestion_levels[i] = rng.choice(VALID_CONGESTION_LEVELS)

    df = pd.DataFrame({
        "distance_km": distance_km,
        "scheduled_hour": scheduled_hour,
        "day_of_week": day_of_week,
        "rainfall_mm": rainfall_mm,
        "prev_station_delay_min": prev_delay_min,
        "route_congestion_index": route_congestion,
        "train_type": train_types,
        DELAY_TARGET_COLUMN: delay_minutes,
        CONGESTION_TARGET_COLUMN: congestion_levels,
    })

    return df


def generate_and_save_data(output_dir: Path = DATA_DIR) -> None:
    """Generate and persist baseline and drifted datasets for delay module."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = generate_delay_dataset(n_samples=2500, drifted=False, seed=101)
    baseline_path = output_dir / "delay_baseline.csv"
    baseline_df.to_csv(baseline_path, index=False)

    drifted_df = generate_delay_dataset(n_samples=1200, drifted=True, seed=202)
    drifted_path = output_dir / "delay_drifted.csv"
    drifted_df.to_csv(drifted_path, index=False)


if __name__ == "__main__":
    generate_and_save_data()
    print("Generated delay_baseline.csv and delay_drifted.csv successfully.")

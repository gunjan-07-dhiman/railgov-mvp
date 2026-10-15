"""
data/generate_fraud_data.py
Synthetic session dataset generator for Fraud Shield & Bot Detection Module.

Privacy & Governance Guarantee:
This module relies strictly on behavioral aggregation and timing features
(counts, velocities, and latency intervals). No raw IP addresses, MAC addresses,
or persistent device hardware identifiers are collected or stored anywhere.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from app.config import (
    DATA_DIR,
    FRAUD_BASELINE_PATH,
    FRAUD_DRIFTED_PATH,
    FRAUD_FEATURE_COLUMNS,
    FRAUD_TARGET_COLUMN,
)


def generate_fraud_dataset(
    n_samples: int = 2000,
    fraud_rate: float = 0.08,
    is_drifted: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic session-level booking interaction dataset.

    Parameters:
    - n_samples: Total number of sessions (default: 2000).
    - fraud_rate: Positive label proportion (default: ~8%).
    - is_drifted: If True, simulates adaptive adversaries and production track shifts.
    - random_seed: Seed for reproducible generation.
    """
    rng = np.random.RandomState(random_seed)
    n_fraud = int(round(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    # --- 1. LEGITIMATE PASSENGER SESSIONS ---
    if not is_drifted:
        # Baseline legitimate behavior: Desktop/web portal distribution
        legit_bookings = rng.poisson(lam=0.5, size=n_legit)
        legit_bookings = np.clip(legit_bookings, 0, 4)

        legit_dev_accs = rng.choice([1, 2, 3], size=n_legit, p=[0.90, 0.08, 0.02])
        legit_ip_accs = rng.choice([1, 2, 3, 4], size=n_legit, p=[0.85, 0.11, 0.03, 0.01])

        # Form fill duration (lognormal distribution around 34s)
        legit_form_fill = rng.lognormal(mean=3.5, sigma=0.55, size=n_legit)
        legit_form_fill = np.clip(legit_form_fill, 6.0, 300.0)

        # Tatkal window proportion (baseline 22%)
        legit_tatkal = rng.binomial(n=1, p=0.22, size=n_legit)

        # Click gap latency (baseline mean ~1400ms)
        legit_click_gap = rng.normal(loc=1400.0, scale=350.0, size=n_legit)
        legit_click_gap = np.clip(legit_click_gap, 550.0, 3500.0)

        # Overlapping hard cases: 4.5% of genuine users are booking agents, power users, or family kiosks
        # with fast autofill, multiple bookings, and shared device profiles
        power_user_idx = rng.choice(n_legit, size=int(0.045 * n_legit), replace=False)
        legit_form_fill[power_user_idx] = rng.uniform(2.5, 4.8, size=len(power_user_idx))
        legit_dev_accs[power_user_idx] = rng.choice([2, 3, 4], size=len(power_user_idx), p=[0.40, 0.40, 0.20])
        legit_ip_accs[power_user_idx] = rng.choice([2, 3, 4], size=len(power_user_idx), p=[0.40, 0.40, 0.20])
        legit_bookings[power_user_idx] = rng.choice([2, 3, 4], size=len(power_user_idx), p=[0.40, 0.40, 0.20])
        legit_click_gap[power_user_idx] = rng.uniform(180.0, 380.0, size=len(power_user_idx))
        legit_tatkal[power_user_idx] = 1

    else:
        # Drifted legitimate behavior: Shift toward mobile app / peak holiday traffic
        legit_bookings = rng.poisson(lam=0.9, size=n_legit)
        legit_bookings = np.clip(legit_bookings, 0, 5)

        legit_dev_accs = rng.choice([1, 2, 3], size=n_legit, p=[0.82, 0.14, 0.04])
        legit_ip_accs = rng.choice([1, 2, 3, 4], size=n_legit, p=[0.75, 0.18, 0.05, 0.02])

        # Mobile app autofill speeds up genuine checkouts (median ~19s) -> Covariate shift
        legit_form_fill = rng.lognormal(mean=2.9, sigma=0.50, size=n_legit)
        legit_form_fill = np.clip(legit_form_fill, 4.0, 200.0)

        # Peak holiday Tatkal surge: Tatkal proportion rises to 44% -> Covariate shift
        legit_tatkal = rng.binomial(n=1, p=0.44, size=n_legit)

        # Mobile touch interaction gap is faster (mean ~920ms) -> Covariate shift
        legit_click_gap = rng.normal(loc=920.0, scale=280.0, size=n_legit)
        legit_click_gap = np.clip(legit_click_gap, 350.0, 2400.0)

    # Failed payments & account age
    legit_failed_pay = rng.choice([0, 1, 2], size=n_legit, p=[0.88, 0.10, 0.02])
    legit_age_days = rng.exponential(scale=240, size=n_legit) + 15
    legit_age_days = np.clip(legit_age_days, 1, 1500).astype(int)

    # --- 2. FRAUD / BOT SESSIONS ---
    if not is_drifted:
        # BASELINE ATTACKERS: High-velocity automated scripts
        fraud_bookings = rng.poisson(lam=6.5, size=n_fraud)
        fraud_bookings = np.clip(fraud_bookings, 2, 18)

        fraud_dev_accs = rng.choice([3, 4, 5, 6, 7, 8], size=n_fraud, p=[0.25, 0.30, 0.20, 0.12, 0.08, 0.05])
        fraud_ip_accs = rng.choice([2, 3, 4, 5, 6], size=n_fraud, p=[0.20, 0.35, 0.25, 0.12, 0.08])

        # Sub-second to rapid script form fill (mean 1.8s)
        fraud_form_fill = rng.lognormal(mean=0.6, sigma=0.45, size=n_fraud)
        fraud_form_fill = np.clip(fraud_form_fill, 0.6, 4.5)

        # Concentrated on Tatkal opening quota
        fraud_tatkal = rng.binomial(n=1, p=0.88, size=n_fraud)

        # Mechanical click intervals (mean 140ms)
        fraud_click_gap = rng.normal(loc=140.0, scale=35.0, size=n_fraud)
        fraud_click_gap = np.clip(fraud_click_gap, 65.0, 320.0)

        # Hard case: 18% of bots employ stealth delay to mimic slower humans
        slow_bot_idx = rng.choice(n_fraud, size=int(0.18 * n_fraud), replace=False)
        fraud_form_fill[slow_bot_idx] = rng.uniform(12.0, 28.0, size=len(slow_bot_idx))
        fraud_dev_accs[slow_bot_idx] = rng.choice([1, 2], size=len(slow_bot_idx), p=[0.70, 0.30])
        fraud_ip_accs[slow_bot_idx] = rng.choice([1, 2], size=len(slow_bot_idx), p=[0.60, 0.40])
        fraud_bookings[slow_bot_idx] = rng.choice([1, 2], size=len(slow_bot_idx), p=[0.60, 0.40])
        fraud_click_gap[slow_bot_idx] = rng.uniform(700.0, 1300.0, size=len(slow_bot_idx))

        fraud_failed_pay = rng.choice([0, 1, 2, 3, 4], size=n_fraud, p=[0.20, 0.35, 0.25, 0.14, 0.06])
        fraud_age_days = rng.exponential(scale=6, size=n_fraud)
        fraud_age_days = np.clip(fraud_age_days, 0, 30).astype(int)

    else:
        # DRIFTED ATTACKERS: Sophisticated evasion tactics
        # Attackers rotate residential proxies and spoof browser fingerprints
        fraud_bookings = rng.poisson(lam=1.8, size=n_fraud)
        fraud_bookings = np.clip(fraud_bookings, 1, 3)

        # Fingerprint spoofing lowers device and IP account concentrations
        fraud_dev_accs = rng.choice([1, 2], size=n_fraud, p=[0.85, 0.15])
        fraud_ip_accs = rng.choice([1, 2], size=n_fraud, p=[0.80, 0.20])

        # Human-like delay injections to bypass fast-fill rules (mean 16s)
        fraud_form_fill = rng.lognormal(mean=2.8, sigma=0.40, size=n_fraud)
        fraud_form_fill = np.clip(fraud_form_fill, 9.0, 35.0)

        fraud_failed_pay = rng.choice([0, 1, 2, 3], size=n_fraud, p=[0.35, 0.40, 0.18, 0.07])
        fraud_age_days = rng.exponential(scale=18, size=n_fraud)
        fraud_age_days = np.clip(fraud_age_days, 1, 90).astype(int)

        # Still heavily targeted at Tatkal opening quota
        fraud_tatkal = rng.binomial(n=1, p=0.86, size=n_fraud)

        # Randomized click delays imitating human mouse intervals (mean 900ms)
        fraud_click_gap = rng.normal(loc=900.0, scale=220.0, size=n_fraud)
        fraud_click_gap = np.clip(fraud_click_gap, 550.0, 1800.0)

    # --- 3. ASSEMBLE DATAFRAME ---
    bookings = np.concatenate([legit_bookings, fraud_bookings])
    dev_accs = np.concatenate([legit_dev_accs, fraud_dev_accs])
    ip_accs = np.concatenate([legit_ip_accs, fraud_ip_accs])
    form_fills = np.round(np.concatenate([legit_form_fill, fraud_form_fill]), 2)
    failed_pays = np.concatenate([legit_failed_pay, fraud_failed_pay])
    ages = np.concatenate([legit_age_days, fraud_age_days])
    tatkals = np.concatenate([legit_tatkal, fraud_tatkal])
    click_gaps = np.round(np.concatenate([legit_click_gap, fraud_click_gap]), 1)
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame({
        "bookings_last_hour": bookings,
        "accounts_per_device": dev_accs,
        "accounts_per_ip": ip_accs,
        "form_fill_seconds": form_fills,
        "failed_payments": failed_pays,
        "account_age_days": ages,
        "is_tatkal_window": tatkals,
        "avg_gap_between_clicks_ms": click_gaps,
        FRAUD_TARGET_COLUMN: labels,
    })

    shuffle_idx = rng.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    return df


def generate_and_save_all(
    baseline_path: Path = FRAUD_BASELINE_PATH,
    drifted_path: Path = FRAUD_DRIFTED_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and write both baseline and drifted fraud datasets to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_df = generate_fraud_dataset(n_samples=2000, fraud_rate=0.08, is_drifted=False, random_seed=42)
    drift_df = generate_fraud_dataset(n_samples=2000, fraud_rate=0.08, is_drifted=True, random_seed=101)

    base_df.to_csv(baseline_path, index=False)
    drift_df.to_csv(drifted_path, index=False)

    return base_df, drift_df


if __name__ == "__main__":
    b_df, d_df = generate_and_save_all()
    print(f"Generated fraud baseline: {b_df.shape}, fraud rate: {b_df['is_fraud'].mean():.3%}")
    print(f"Generated fraud drifted: {d_df.shape}, fraud rate: {d_df['is_fraud'].mean():.3%}")

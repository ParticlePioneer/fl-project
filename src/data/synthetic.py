"""
Generates a synthetic dataset that mimics the *shape* of the cleaned
LendingClub dataset (same column names/types as configs/base.yaml expects),
so the baseline model, partitioning, and training pipeline can all be built
and tested before the real cleaned CSV is delivered.

Delete or ignore this once data/processed/lendingclub_clean.csv exists --
just point configs/base.yaml at the real file and everything downstream
should work unchanged, since column names match.
"""

import numpy as np
import pandas as pd

STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"]


def generate_synthetic_lendingclub(n_rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_inc = rng.lognormal(mean=10.5, sigma=0.6, size=n_rows).clip(15000, 300000)
    loan_amnt = rng.uniform(1000, 40000, size=n_rows)
    int_rate = rng.uniform(5, 30, size=n_rows)
    dti = rng.uniform(0, 40, size=n_rows)
    emp_length = rng.integers(0, 11, size=n_rows)
    addr_state = rng.choice(STATES, size=n_rows, p=_state_weights())
    purpose = rng.choice(PURPOSES, size=n_rows)

    # Synthetic default probability -- higher int_rate/dti, lower income => more likely default.
    default_logit = (
        0.05 * int_rate + 0.03 * dti - 0.00002 * annual_inc - 0.00005 * loan_amnt - 2.0
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    loan_status = (rng.uniform(size=n_rows) < default_prob).astype(int)  # 1 = default

    df = pd.DataFrame({
        "annual_inc": annual_inc,
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "dti": dti,
        "emp_length": emp_length,
        "addr_state": addr_state,
        "purpose": purpose,
        "loan_status": loan_status,
    })

    # income bracket for fairness analysis, matches configs/base.yaml fairness_column
    df["annual_inc_bracket"] = pd.cut(
        df["annual_inc"],
        bins=[0, 40000, 80000, 150000, np.inf],
        labels=["low", "mid", "high", "very_high"],
    )

    return df


def _state_weights():
    """Deliberately skewed state distribution -- mimics real regional volume
    imbalance so the non-IID partitioning has something real to bite into,
    even in synthetic data."""
    weights = np.array([25, 18, 15, 12, 8, 7, 6, 4, 3, 2], dtype=float)
    return weights / weights.sum()


if __name__ == "__main__":
    df = generate_synthetic_lendingclub()
    df.to_csv("data/processed/synthetic_lendingclub.csv", index=False)
    print(f"Generated {len(df)} synthetic rows -> data/processed/synthetic_lendingclub.csv")
    print(df["loan_status"].value_counts(normalize=True))
    print(df["addr_state"].value_counts(normalize=True))
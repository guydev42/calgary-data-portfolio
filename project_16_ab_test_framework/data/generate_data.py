"""Generate synthetic A/B test experiment data."""

import numpy as np
import pandas as pd
import os


def generate_ab_test_data(seed=42):
    """Generate 3 experiments with 5,000 users per variant each (30,000 rows total)."""
    rng = np.random.RandomState(seed)
    n_per_variant = 5000

    experiments = []

    # Experiment 1: Website redesign (significant)
    # control: 3.2% conversion, treatment: 3.8%
    for variant, rate in [("control", 0.030), ("treatment", 0.042)]:
        user_ids = [f"user_{i:06d}" for i in range(
            len(experiments), len(experiments) + n_per_variant
        )]
        converted = rng.binomial(1, rate, n_per_variant)
        # Revenue: $0 if not converted, lognormal if converted
        revenue = np.where(
            converted == 1,
            rng.lognormal(3.5, 0.8, n_per_variant).round(2),
            0.0
        )
        session_dur = rng.lognormal(1.8, 0.5, n_per_variant).clip(0.5, 60).round(2)
        if variant == "treatment":
            session_dur *= 1.05  # slight increase in engagement
        pages = rng.poisson(4.5 if variant == "control" else 5.0, n_per_variant).clip(1, 25)

        exp_df = pd.DataFrame({
            "experiment_id": "exp_001",
            "variant": variant,
            "user_id": user_ids,
            "converted": converted,
            "revenue": revenue.round(2),
            "session_duration_min": session_dur.round(2),
            "pages_viewed": pages,
        })
        experiments.append(exp_df)

    # Experiment 2: Pricing change (NOT significant)
    # control: 2.1%, treatment: 2.3%
    for variant, rate in [("control", 0.021), ("treatment", 0.023)]:
        user_ids = [f"user_{i:06d}" for i in range(
            sum(len(e) for e in experiments),
            sum(len(e) for e in experiments) + n_per_variant
        )]
        converted = rng.binomial(1, rate, n_per_variant)
        revenue = np.where(
            converted == 1,
            rng.lognormal(4.0, 0.6, n_per_variant).round(2),
            0.0
        )
        session_dur = rng.lognormal(1.6, 0.4, n_per_variant).clip(0.5, 45).round(2)
        pages = rng.poisson(3.8, n_per_variant).clip(1, 20)

        exp_df = pd.DataFrame({
            "experiment_id": "exp_002",
            "variant": variant,
            "user_id": user_ids,
            "converted": converted,
            "revenue": revenue.round(2),
            "session_duration_min": session_dur.round(2),
            "pages_viewed": pages,
        })
        experiments.append(exp_df)

    # Experiment 3: Email subject line (significant)
    # control: 12%, treatment: 15%
    for variant, rate in [("control", 0.12), ("treatment", 0.15)]:
        user_ids = [f"user_{i:06d}" for i in range(
            sum(len(e) for e in experiments),
            sum(len(e) for e in experiments) + n_per_variant
        )]
        converted = rng.binomial(1, rate, n_per_variant)
        revenue = np.where(
            converted == 1,
            rng.lognormal(3.0, 0.7, n_per_variant).round(2),
            0.0
        )
        session_dur = rng.lognormal(1.5, 0.6, n_per_variant).clip(0.3, 40).round(2)
        pages = rng.poisson(3.0, n_per_variant).clip(1, 15)

        exp_df = pd.DataFrame({
            "experiment_id": "exp_003",
            "variant": variant,
            "user_id": user_ids,
            "converted": converted,
            "revenue": revenue.round(2),
            "session_duration_min": session_dur.round(2),
            "pages_viewed": pages,
        })
        experiments.append(exp_df)

    df = pd.concat(experiments, ignore_index=True)

    # Summary
    for exp_id in ["exp_001", "exp_002", "exp_003"]:
        exp = df[df["experiment_id"] == exp_id]
        for var in ["control", "treatment"]:
            subset = exp[exp["variant"] == var]
            rate = subset["converted"].mean()
            print(f"  {exp_id} / {var}: n={len(subset)}, conversion={rate:.3%}")

    out_path = os.path.join(os.path.dirname(__file__), "ab_test_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    generate_ab_test_data()

"""Core A/B test analysis module with frequentist, Bayesian, and sequential methods."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def power_analysis(baseline_rate, mde, alpha=0.05, power=0.80):
    """Compute required sample size per group for a two-proportion z-test.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate for the control group.
    mde : float
        Minimum detectable effect (absolute difference).
    alpha : float
        Significance level (default 0.05).
    power : float
        Statistical power (default 0.80).

    Returns
    -------
    dict with sample_size_per_group, total_sample_size, and input params.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = (
        (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
         + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / mde ** 2
    )
    n = int(np.ceil(n))

    return {
        "sample_size_per_group": n,
        "total_sample_size": 2 * n,
        "baseline_rate": baseline_rate,
        "mde": mde,
        "alpha": alpha,
        "power": power,
    }


# ---------------------------------------------------------------------------
# Frequentist tests
# ---------------------------------------------------------------------------

def frequentist_test(control_data, treatment_data, metric="conversion", alpha=0.05):
    """Run frequentist hypothesis test.

    Parameters
    ----------
    control_data : array-like
        Values for the control group.
    treatment_data : array-like
        Values for the treatment group.
    metric : str
        'conversion' for binary (chi-squared), 'continuous' for t-test.
    alpha : float
        Significance level.

    Returns
    -------
    dict with p_value, effect_size, ci_lower, ci_upper, significant, test_used.
    """
    control = np.asarray(control_data)
    treatment = np.asarray(treatment_data)

    if metric == "conversion":
        n_c, n_t = len(control), len(treatment)
        p_c = control.mean()
        p_t = treatment.mean()
        effect = p_t - p_c

        # Pooled proportion
        p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))

        if se == 0:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = effect / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # CI for the difference
        se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = effect - z_crit * se_diff
        ci_upper = effect + z_crit * se_diff

        # Cohen's h for proportions
        effect_size = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))
        test_used = "z-test for proportions"

    else:
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        effect = treatment.mean() - control.mean()

        # Pooled SE for CI
        se_diff = np.sqrt(treatment.var(ddof=1)/len(treatment) + control.var(ddof=1)/len(control))
        df_welch = (
            (treatment.var(ddof=1)/len(treatment) + control.var(ddof=1)/len(control))**2
            / (
                (treatment.var(ddof=1)/len(treatment))**2 / (len(treatment)-1)
                + (control.var(ddof=1)/len(control))**2 / (len(control)-1)
            )
        )
        t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
        ci_lower = effect - t_crit * se_diff
        ci_upper = effect + t_crit * se_diff

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(control)-1)*control.var(ddof=1) + (len(treatment)-1)*treatment.var(ddof=1))
            / (len(control) + len(treatment) - 2)
        )
        effect_size = effect / pooled_std if pooled_std > 0 else 0.0
        test_used = "Welch's t-test"

    return {
        "p_value": float(p_value),
        "effect": float(effect),
        "effect_size": float(effect_size),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "test_used": test_used,
        "n_control": len(control),
        "n_treatment": len(treatment),
        "mean_control": float(control.mean()),
        "mean_treatment": float(treatment.mean()),
    }


# ---------------------------------------------------------------------------
# Bayesian A/B test
# ---------------------------------------------------------------------------

def bayesian_ab(control_data, treatment_data, n_samples=100000, seed=42):
    """Bayesian A/B test using Beta-Binomial model for conversion data.

    Uses uninformative Beta(1, 1) prior.

    Returns
    -------
    dict with prob_treatment_better, expected_lift, credible_interval,
    posterior samples for both groups.
    """
    rng = np.random.RandomState(seed)
    control = np.asarray(control_data)
    treatment = np.asarray(treatment_data)

    # Posterior parameters (Beta-Binomial)
    alpha_c = 1 + control.sum()
    beta_c = 1 + len(control) - control.sum()
    alpha_t = 1 + treatment.sum()
    beta_t = 1 + len(treatment) - treatment.sum()

    # Draw posterior samples
    samples_c = rng.beta(alpha_c, beta_c, n_samples)
    samples_t = rng.beta(alpha_t, beta_t, n_samples)

    # P(treatment > control)
    prob_better = (samples_t > samples_c).mean()

    # Lift distribution
    lift = (samples_t - samples_c) / samples_c
    expected_lift = lift.mean()

    # 95% HDI for lift
    lift_sorted = np.sort(lift)
    ci_lower = float(np.percentile(lift, 2.5))
    ci_upper = float(np.percentile(lift, 97.5))

    # Credible intervals for each group
    ci_control = (float(np.percentile(samples_c, 2.5)), float(np.percentile(samples_c, 97.5)))
    ci_treatment = (float(np.percentile(samples_t, 2.5)), float(np.percentile(samples_t, 97.5)))

    return {
        "prob_treatment_better": float(prob_better),
        "expected_lift": float(expected_lift),
        "lift_ci_lower": ci_lower,
        "lift_ci_upper": ci_upper,
        "control_posterior": samples_c,
        "treatment_posterior": samples_t,
        "lift_distribution": lift,
        "control_ci": ci_control,
        "treatment_ci": ci_treatment,
        "control_mean": float(samples_c.mean()),
        "treatment_mean": float(samples_t.mean()),
    }


# ---------------------------------------------------------------------------
# Sequential testing
# ---------------------------------------------------------------------------

def sequential_test(data, alpha=0.05, beta=0.20, n_looks=5):
    """O'Brien-Fleming-style group sequential test.

    Simulates interim analyses at equal intervals using
    an O'Brien-Fleming spending function.

    Parameters
    ----------
    data : DataFrame
        Must have 'variant' and 'converted' columns.
    alpha : float
        Overall significance level.
    beta : float
        Type II error rate.
    n_looks : int
        Number of interim analyses.

    Returns
    -------
    dict with boundaries, z_stats at each look, and early-stop decision.
    """
    control = data[data["variant"] == "control"]["converted"].values
    treatment = data[data["variant"] == "treatment"]["converted"].values

    n_c = len(control)
    n_t = len(treatment)
    n_min = min(n_c, n_t)

    look_sizes = np.linspace(n_min // n_looks, n_min, n_looks).astype(int)

    results = []
    stopped_early = False
    stop_at_look = None

    for i, look_n in enumerate(look_sizes):
        info_fraction = (i + 1) / n_looks

        # O'Brien-Fleming boundary (approximate)
        z_boundary = stats.norm.ppf(1 - alpha / 2) / np.sqrt(info_fraction)

        c_subset = control[:look_n]
        t_subset = treatment[:look_n]

        p_c = c_subset.mean()
        p_t = t_subset.mean()
        p_pool = (c_subset.sum() + t_subset.sum()) / (2 * look_n)
        se = np.sqrt(2 * p_pool * (1 - p_pool) / look_n) if p_pool > 0 and p_pool < 1 else 1e-10

        z_stat = (p_t - p_c) / se if se > 0 else 0.0

        reject = abs(z_stat) > z_boundary

        results.append({
            "look": i + 1,
            "n_per_group": int(look_n),
            "info_fraction": round(info_fraction, 3),
            "z_boundary": round(z_boundary, 4),
            "z_stat": round(z_stat, 4),
            "p_control": round(p_c, 5),
            "p_treatment": round(p_t, 5),
            "reject": reject,
        })

        if reject and not stopped_early:
            stopped_early = True
            stop_at_look = i + 1

    return {
        "results": results,
        "stopped_early": stopped_early,
        "stop_at_look": stop_at_look,
        "n_looks": n_looks,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def multiple_comparison_correction(p_values, method="bonferroni"):
    """Apply multiple comparison correction to a list of p-values.

    Methods: bonferroni, holm, bh_fdr.

    Returns
    -------
    dict with original p-values, adjusted p-values, and rejection decisions.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    if method == "bonferroni":
        adjusted = np.minimum(p * m, 1.0)

    elif method == "holm":
        order = np.argsort(p)
        adjusted = np.zeros(m)
        for i, idx in enumerate(order):
            adjusted[idx] = p[idx] * (m - i)
        # Enforce monotonicity
        for i in range(1, m):
            idx = order[i]
            prev_idx = order[i - 1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
        adjusted = np.minimum(adjusted, 1.0)

    elif method == "bh_fdr":
        order = np.argsort(p)
        adjusted = np.zeros(m)
        for i, idx in enumerate(order):
            adjusted[idx] = p[idx] * m / (i + 1)
        # Enforce monotonicity (reverse)
        for i in range(m - 2, -1, -1):
            idx = order[i]
            next_idx = order[i + 1]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx])
        adjusted = np.minimum(adjusted, 1.0)

    else:
        raise ValueError(f"Unknown method: {method}. Use bonferroni, holm, or bh_fdr.")

    return {
        "method": method,
        "original_p_values": p.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "rejected": (adjusted < 0.05).tolist(),
        "n_rejected": int((adjusted < 0.05).sum()),
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def summary_report(experiment_data, experiment_id=None):
    """Generate a structured summary for an experiment.

    Parameters
    ----------
    experiment_data : DataFrame
        Must have variant, converted, revenue, session_duration_min, pages_viewed columns.
    experiment_id : str, optional
        Identifier for the experiment.

    Returns
    -------
    dict with all analysis results.
    """
    control = experiment_data[experiment_data["variant"] == "control"]
    treatment = experiment_data[experiment_data["variant"] == "treatment"]

    # Conversion test
    freq_conversion = frequentist_test(
        control["converted"].values,
        treatment["converted"].values,
        metric="conversion"
    )

    # Revenue test (continuous)
    freq_revenue = frequentist_test(
        control["revenue"].values,
        treatment["revenue"].values,
        metric="continuous"
    )

    # Bayesian conversion
    bayes = bayesian_ab(
        control["converted"].values,
        treatment["converted"].values
    )

    # Sequential test
    seq = sequential_test(experiment_data)

    report = {
        "experiment_id": experiment_id,
        "n_control": len(control),
        "n_treatment": len(treatment),
        "control_conversion_rate": float(control["converted"].mean()),
        "treatment_conversion_rate": float(treatment["converted"].mean()),
        "control_avg_revenue": float(control["revenue"].mean()),
        "treatment_avg_revenue": float(treatment["revenue"].mean()),
        "frequentist_conversion": freq_conversion,
        "frequentist_revenue": freq_revenue,
        "bayesian": {
            "prob_treatment_better": bayes["prob_treatment_better"],
            "expected_lift": bayes["expected_lift"],
            "lift_ci": (bayes["lift_ci_lower"], bayes["lift_ci_upper"]),
        },
        "sequential": {
            "stopped_early": seq["stopped_early"],
            "stop_at_look": seq["stop_at_look"],
        },
    }

    return report


def print_summary(report):
    """Pretty-print an experiment summary."""
    print(f"\nExperiment: {report['experiment_id']}")
    print("=" * 55)
    print(f"  Control:   n={report['n_control']:,}, "
          f"conversion={report['control_conversion_rate']:.3%}, "
          f"avg revenue=${report['control_avg_revenue']:.2f}")
    print(f"  Treatment: n={report['n_treatment']:,}, "
          f"conversion={report['treatment_conversion_rate']:.3%}, "
          f"avg revenue=${report['treatment_avg_revenue']:.2f}")

    fc = report["frequentist_conversion"]
    print(f"\n  Frequentist (conversion):")
    print(f"    p-value:     {fc['p_value']:.4f}")
    print(f"    Effect:      {fc['effect']:+.4f}")
    print(f"    95% CI:      [{fc['ci_lower']:.4f}, {fc['ci_upper']:.4f}]")
    print(f"    Significant: {'Yes' if fc['significant'] else 'No'}")

    b = report["bayesian"]
    print(f"\n  Bayesian:")
    print(f"    P(B > A):    {b['prob_treatment_better']:.3f}")
    print(f"    Expected lift: {b['expected_lift']:.2%}")
    print(f"    95% CI lift: [{b['lift_ci'][0]:.2%}, {b['lift_ci'][1]:.2%}]")

    s = report["sequential"]
    if s["stopped_early"]:
        print(f"\n  Sequential: early stop at look {s['stop_at_look']}")
    else:
        print(f"\n  Sequential: no early stopping triggered")
    print("=" * 55)

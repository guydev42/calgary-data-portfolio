"""Visualization functions for A/B test analysis."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_conversion_bar(control_rate, treatment_rate, control_n, treatment_n,
                        experiment_id="", alpha=0.05, save_path=None):
    """Bar chart of conversion rates with error bars."""
    fig, ax = plt.subplots(figsize=(6, 5))

    rates = [control_rate, treatment_rate]
    labels = ["Control", "Treatment"]
    colors = ["#90CAF9", "#2196F3"]

    # Standard error for proportions
    se_c = np.sqrt(control_rate * (1 - control_rate) / control_n)
    se_t = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment_n)
    z = stats.norm.ppf(1 - alpha / 2)
    errors = [z * se_c, z * se_t]

    bars = ax.bar(labels, rates, color=colors, yerr=errors, capsize=8,
                  edgecolor="gray", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors) * 0.3,
                f"{rate:.2%}", ha="center", fontweight="bold", fontsize=12)

    ax.set_ylabel("Conversion rate")
    ax.set_title(f"Conversion rate comparison{' - ' + experiment_id if experiment_id else ''}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(rates) * 1.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_posterior_distributions(bayes_result, experiment_id="", save_path=None):
    """Overlay posterior distributions for control and treatment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: conversion rate posteriors
    ax = axes[0]
    samples_c = bayes_result["control_posterior"]
    samples_t = bayes_result["treatment_posterior"]

    ax.hist(samples_c, bins=80, alpha=0.5, density=True, color="#90CAF9", label="Control")
    ax.hist(samples_t, bins=80, alpha=0.5, density=True, color="#1565C0", label="Treatment")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior distributions{' - ' + experiment_id if experiment_id else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: lift distribution
    ax = axes[1]
    lift = bayes_result["lift_distribution"]
    ax.hist(lift, bins=80, alpha=0.7, density=True, color="#2196F3")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No effect")
    ax.axvline(np.mean(lift), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean lift: {np.mean(lift):.2%}")
    ax.set_xlabel("Relative lift")
    ax.set_ylabel("Density")
    ax.set_title("Lift distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_sequential_monitoring(seq_result, experiment_id="", save_path=None):
    """Sequential monitoring chart with cumulative effect and decision boundaries."""
    fig, ax = plt.subplots(figsize=(8, 5))

    results = seq_result["results"]
    looks = [r["look"] for r in results]
    z_stats = [r["z_stat"] for r in results]
    boundaries_upper = [r["z_boundary"] for r in results]
    boundaries_lower = [-r["z_boundary"] for r in results]

    ax.plot(looks, z_stats, "b-o", linewidth=2, markersize=8, label="Observed z-statistic")
    ax.plot(looks, boundaries_upper, "r--", linewidth=1.5, label="O'Brien-Fleming boundary")
    ax.plot(looks, boundaries_lower, "r--", linewidth=1.5)
    ax.fill_between(looks, boundaries_lower, boundaries_upper, alpha=0.08, color="red")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)

    # Mark the stop point if early stopping occurred
    if seq_result["stopped_early"]:
        stop = seq_result["stop_at_look"] - 1
        ax.plot(looks[stop], z_stats[stop], "r*", markersize=15, zorder=5,
                label=f"Early stop (look {seq_result['stop_at_look']})")

    ax.set_xlabel("Interim look")
    ax.set_ylabel("Z-statistic")
    ax.set_title(f"Sequential monitoring{' - ' + experiment_id if experiment_id else ''}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(looks)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_sample_size_vs_mde(baseline_rate=0.03, alpha=0.05, power=0.80,
                             mde_range=None, save_path=None):
    """Sample size vs minimum detectable effect curve."""
    if mde_range is None:
        mde_range = np.arange(0.002, 0.05, 0.001)

    fig, ax = plt.subplots(figsize=(8, 5))

    sample_sizes = []
    for mde in mde_range:
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2

        z_a = stats.norm.ppf(1 - alpha / 2)
        z_b = stats.norm.ppf(power)

        n = (
            (z_a * np.sqrt(2 * p_avg * (1 - p_avg))
             + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            / mde ** 2
        )
        sample_sizes.append(int(np.ceil(n)))

    ax.plot(mde_range * 100, sample_sizes, "b-", linewidth=2)
    ax.set_xlabel("Minimum detectable effect (percentage points)")
    ax.set_ylabel("Required sample size per group")
    ax.set_title(f"Sample size vs MDE (baseline={baseline_rate:.1%}, power={power:.0%})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Annotate a key point
    mid = len(mde_range) // 2
    ax.annotate(f"n={sample_sizes[mid]:,}\nat MDE={mde_range[mid]*100:.1f}pp",
                xy=(mde_range[mid]*100, sample_sizes[mid]),
                xytext=(mde_range[mid]*100 + 0.5, sample_sizes[mid] * 1.3),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_forest(experiments_summary, save_path=None):
    """Forest plot for multiple experiments showing effect sizes with CIs."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(experiments_summary) * 1.2)))

    y_positions = list(range(len(experiments_summary)))
    labels = []
    effects = []
    ci_lowers = []
    ci_uppers = []
    colors = []

    for exp in experiments_summary:
        fc = exp["frequentist_conversion"]
        labels.append(exp["experiment_id"])
        effects.append(fc["effect"])
        ci_lowers.append(fc["ci_lower"])
        ci_uppers.append(fc["ci_upper"])
        colors.append("#2196F3" if fc["significant"] else "#90CAF9")

    for i, (eff, lo, hi, color) in enumerate(zip(effects, ci_lowers, ci_uppers, colors)):
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.plot(eff, i, "o", color=color, markersize=10)

    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect size (treatment - control)")
    ax.set_title("Forest plot of experiment effects")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

"""Streamlit dashboard for A/B test analysis framework."""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.experiment import (
    frequentist_test, bayesian_ab, sequential_test,
    multiple_comparison_correction, power_analysis,
    summary_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="A/B test framework", layout="wide")

EXPERIMENT_LABELS = {
    "exp_001": "Website redesign",
    "exp_002": "Pricing change",
    "exp_003": "Email subject line",
}


@st.cache_data
def load_data():
    path = os.path.join(PROJECT_DIR, "data", "ab_test_results.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Frequentist", "Bayesian", "Power calculator", "Report"],
)

df = load_data()
experiments = sorted(df["experiment_id"].unique())


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("A/B test analysis framework")
    st.markdown("Summary of all experiments with key metrics and significance.")

    rows = []
    for exp_id in experiments:
        exp = df[df["experiment_id"] == exp_id]
        ctrl = exp[exp["variant"] == "control"]
        trt = exp[exp["variant"] == "treatment"]

        freq = frequentist_test(
            ctrl["converted"].values, trt["converted"].values, metric="conversion"
        )

        rows.append({
            "Experiment": exp_id,
            "Description": EXPERIMENT_LABELS.get(exp_id, ""),
            "N per group": len(ctrl),
            "Control rate": f"{ctrl['converted'].mean():.2%}",
            "Treatment rate": f"{trt['converted'].mean():.2%}",
            "Effect": f"{freq['effect']:+.4f}",
            "p-value": f"{freq['p_value']:.4f}",
            "Significant": "Yes" if freq["significant"] else "No",
            "Effect size (h)": f"{freq['effect_size']:.4f}",
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Quick visual
    st.subheader("Conversion rates across experiments")
    plot_data = []
    for exp_id in experiments:
        exp = df[df["experiment_id"] == exp_id]
        for var in ["control", "treatment"]:
            subset = exp[exp["variant"] == var]
            plot_data.append({
                "Experiment": exp_id,
                "Variant": var,
                "Conversion rate": subset["converted"].mean(),
            })
    plot_df = pd.DataFrame(plot_data)

    fig = px.bar(plot_df, x="Experiment", y="Conversion rate", color="Variant",
                 barmode="group",
                 color_discrete_map={"control": "#90CAF9", "treatment": "#1565C0"},
                 title="Conversion rates by experiment and variant")
    fig.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Frequentist
# ---------------------------------------------------------------------------
elif page == "Frequentist":
    st.title("Frequentist analysis")

    selected = st.selectbox(
        "Select experiment",
        experiments,
        format_func=lambda x: f"{x} - {EXPERIMENT_LABELS.get(x, '')}",
    )

    exp = df[df["experiment_id"] == selected]
    ctrl = exp[exp["variant"] == "control"]
    trt = exp[exp["variant"] == "treatment"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conversion test")
        freq_conv = frequentist_test(
            ctrl["converted"].values, trt["converted"].values, metric="conversion"
        )
        st.metric("Control rate", f"{freq_conv['mean_control']:.3%}")
        st.metric("Treatment rate", f"{freq_conv['mean_treatment']:.3%}")
        st.metric("Absolute effect", f"{freq_conv['effect']:+.4f}")
        st.metric("p-value", f"{freq_conv['p_value']:.4f}")
        st.metric("95% CI", f"[{freq_conv['ci_lower']:.4f}, {freq_conv['ci_upper']:.4f}]")

        if freq_conv["significant"]:
            st.success(f"Statistically significant at alpha=0.05")
        else:
            st.warning(f"NOT statistically significant at alpha=0.05")

    with col2:
        st.subheader("Revenue test")
        freq_rev = frequentist_test(
            ctrl["revenue"].values, trt["revenue"].values, metric="continuous"
        )
        st.metric("Control avg revenue", f"${freq_rev['mean_control']:.2f}")
        st.metric("Treatment avg revenue", f"${freq_rev['mean_treatment']:.2f}")
        st.metric("Difference", f"${freq_rev['effect']:+.2f}")
        st.metric("p-value", f"{freq_rev['p_value']:.4f}")
        st.metric("Cohen's d", f"{freq_rev['effect_size']:.4f}")

    # Conversion bar chart with error bars
    st.subheader("Conversion rate comparison")
    cr_c = freq_conv["mean_control"]
    cr_t = freq_conv["mean_treatment"]
    n_c = freq_conv["n_control"]
    n_t = freq_conv["n_treatment"]

    se_c = np.sqrt(cr_c * (1 - cr_c) / n_c)
    se_t = np.sqrt(cr_t * (1 - cr_t) / n_t)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Treatment"], y=[cr_c, cr_t],
        error_y=dict(type="data", array=[1.96 * se_c, 1.96 * se_t]),
        marker_color=["#90CAF9", "#1565C0"],
        text=[f"{cr_c:.2%}", f"{cr_t:.2%}"], textposition="outside",
    ))
    fig.update_layout(yaxis_tickformat=".2%", title=f"Conversion rates - {selected}")
    st.plotly_chart(fig, use_container_width=True)

    # Sequential monitoring
    st.subheader("Sequential monitoring")
    seq = sequential_test(exp)

    seq_df = pd.DataFrame(seq["results"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seq_df["look"], y=seq_df["z_stat"],
        mode="lines+markers", name="Z-statistic",
        line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"], y=seq_df["z_boundary"],
        mode="lines", name="Upper boundary",
        line=dict(color="red", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=seq_df["look"], y=-seq_df["z_boundary"],
        mode="lines", name="Lower boundary",
        line=dict(color="red", dash="dash"),
    ))
    fig.update_layout(title=f"Sequential monitoring - {selected}",
                      xaxis_title="Interim look", yaxis_title="Z-statistic")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Bayesian
# ---------------------------------------------------------------------------
elif page == "Bayesian":
    st.title("Bayesian analysis")

    selected = st.selectbox(
        "Select experiment",
        experiments,
        format_func=lambda x: f"{x} - {EXPERIMENT_LABELS.get(x, '')}",
    )

    exp = df[df["experiment_id"] == selected]
    ctrl = exp[exp["variant"] == "control"]["converted"].values
    trt = exp[exp["variant"] == "treatment"]["converted"].values

    bayes = bayesian_ab(ctrl, trt)

    col1, col2, col3 = st.columns(3)
    col1.metric("P(treatment > control)", f"{bayes['prob_treatment_better']:.3f}")
    col2.metric("Expected lift", f"{bayes['expected_lift']:.2%}")
    col3.metric("95% credible interval",
                f"[{bayes['lift_ci_lower']:.2%}, {bayes['lift_ci_upper']:.2%}]")

    # Posterior distributions
    st.subheader("Posterior distributions")

    fig = go.Figure()
    # Sample down for plotting
    n_plot = 5000
    rng = np.random.RandomState(42)
    idx_c = rng.choice(len(bayes["control_posterior"]), n_plot, replace=False)
    idx_t = rng.choice(len(bayes["treatment_posterior"]), n_plot, replace=False)

    fig.add_trace(go.Histogram(
        x=bayes["control_posterior"][idx_c], nbinsx=80,
        name="Control", opacity=0.5, marker_color="#90CAF9",
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=bayes["treatment_posterior"][idx_t], nbinsx=80,
        name="Treatment", opacity=0.5, marker_color="#1565C0",
        histnorm="probability density",
    ))
    fig.update_layout(
        barmode="overlay", title="Posterior conversion rate distributions",
        xaxis_title="Conversion rate", yaxis_title="Density",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lift distribution
    st.subheader("Lift distribution")
    lift_sample = bayes["lift_distribution"][idx_c]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lift_sample, nbinsx=60, marker_color="#2196F3",
        histnorm="probability density", name="Lift",
    ))
    fig.add_vline(x=0, line_color="red", line_dash="dash", annotation_text="No effect")
    fig.update_layout(
        title="Relative lift distribution",
        xaxis_title="Relative lift", yaxis_title="Density",
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Decision summary
    if bayes["prob_treatment_better"] > 0.95:
        st.success(
            f"Strong evidence that treatment is better "
            f"(P(B>A) = {bayes['prob_treatment_better']:.3f})"
        )
    elif bayes["prob_treatment_better"] > 0.80:
        st.warning(
            f"Moderate evidence that treatment is better "
            f"(P(B>A) = {bayes['prob_treatment_better']:.3f})"
        )
    else:
        st.info(
            f"Insufficient evidence for treatment superiority "
            f"(P(B>A) = {bayes['prob_treatment_better']:.3f})"
        )


# ---------------------------------------------------------------------------
# Page: Power calculator
# ---------------------------------------------------------------------------
elif page == "Power calculator":
    st.title("Power calculator")
    st.markdown("Compute the required sample size for a new experiment.")

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input(
            "Baseline conversion rate", 0.001, 0.99, 0.032,
            step=0.005, format="%.3f"
        )
        mde = st.number_input(
            "Minimum detectable effect (absolute)",
            0.001, 0.50, 0.005, step=0.001, format="%.3f"
        )
    with col2:
        alpha = st.number_input("Significance level (alpha)", 0.01, 0.20, 0.05, step=0.01)
        power_val = st.number_input("Power", 0.50, 0.99, 0.80, step=0.05)

    if st.button("Calculate", type="primary"):
        result = power_analysis(baseline, mde, alpha, power_val)
        col_a, col_b = st.columns(2)
        col_a.metric("Sample size per group", f"{result['sample_size_per_group']:,}")
        col_b.metric("Total sample size", f"{result['total_sample_size']:,}")

    # MDE curve
    st.subheader("Sample size vs minimum detectable effect")
    mde_range = np.arange(0.002, 0.06, 0.001)
    sizes = []
    for m in mde_range:
        r = power_analysis(baseline, m, alpha, power_val)
        sizes.append(r["sample_size_per_group"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mde_range * 100, y=sizes, mode="lines",
        line=dict(color="#2196F3", width=2),
    ))
    fig.update_layout(
        title=f"Required sample size vs MDE (baseline={baseline:.1%})",
        xaxis_title="MDE (percentage points)",
        yaxis_title="Sample size per group",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Report
# ---------------------------------------------------------------------------
elif page == "Report":
    st.title("Experiment summary report")
    st.markdown("Auto-generated summary for stakeholders.")

    reports = []
    p_values = []

    for exp_id in experiments:
        exp = df[df["experiment_id"] == exp_id]
        report = summary_report(exp, exp_id)
        reports.append(report)
        p_values.append(report["frequentist_conversion"]["p_value"])

    # Multiple comparison correction
    correction = multiple_comparison_correction(p_values, method="bh_fdr")

    st.subheader("Results summary")

    for i, (report, adj_p, rejected) in enumerate(zip(
        reports, correction["adjusted_p_values"], correction["rejected"]
    )):
        exp_id = report["experiment_id"]
        label = EXPERIMENT_LABELS.get(exp_id, exp_id)

        with st.expander(f"{exp_id}: {label}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Control rate", f"{report['control_conversion_rate']:.2%}")
            col2.metric("Treatment rate", f"{report['treatment_conversion_rate']:.2%}")

            fc = report["frequentist_conversion"]
            col3.metric("p-value (raw)", f"{fc['p_value']:.4f}")
            col4.metric("p-value (BH-FDR adjusted)", f"{adj_p:.4f}")

            b = report["bayesian"]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("P(treatment > control)", f"{b['prob_treatment_better']:.3f}")
            col_b.metric("Expected lift", f"{b['expected_lift']:.2%}")

            if rejected:
                st.success(f"Conclusion: statistically significant after multiple comparison correction.")
            else:
                st.warning(f"Conclusion: not statistically significant after correction.")

    # Overall recommendation
    st.subheader("Recommendations")
    n_significant = correction["n_rejected"]
    st.markdown(f"**{n_significant} out of {len(experiments)} experiments** showed "
                f"statistically significant results after BH-FDR correction.")

    for i, (report, rejected) in enumerate(zip(reports, correction["rejected"])):
        exp_id = report["experiment_id"]
        label = EXPERIMENT_LABELS.get(exp_id, "")
        if rejected:
            lift = report["bayesian"]["expected_lift"]
            st.markdown(f"- **{exp_id} ({label})**: Implement the treatment. "
                        f"Expected lift: {lift:.1%}")
        else:
            st.markdown(f"- **{exp_id} ({label})**: Insufficient evidence. "
                        f"Consider running longer or accepting current design.")

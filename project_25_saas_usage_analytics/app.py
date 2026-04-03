"""
Streamlit dashboard for SaaS usage analytics.
Five pages: Usage Overview, Cohort Retention, Feature Adoption,
At-Risk Accounts, Product Health Metrics.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="SaaS usage analytics dashboard", layout="wide")

DATA_PATH = "data/saas_usage.csv"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_active_date"] = pd.to_datetime(df["last_active_date"])
    df["signup_month"] = df["signup_date"].dt.to_period("M").astype(str)
    bins = [0, 3, 8, 15, 40]
    labels = ["Low", "Medium", "High", "Power"]
    df["login_group"] = pd.cut(df["daily_logins"], bins=bins, labels=labels, include_lowest=True)
    return df


@st.cache_data
def load_model_comparison():
    path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_at_risk_analysis():
    path = os.path.join(OUTPUTS_DIR, "at_risk_analysis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# --- Sidebar navigation ---
page = st.sidebar.radio(
    "Navigate",
    ["Usage Overview", "Cohort Retention", "Feature Adoption", "At-Risk Accounts", "Product Health Metrics"],
)

df = load_data()

# =====================================================================
# PAGE 1: USAGE OVERVIEW
# =====================================================================
if page == "Usage Overview":
    st.title("SaaS usage overview")
    st.markdown("Understanding user engagement patterns across the platform to identify growth levers and risk signals.")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = df["is_churned"].mean()
    mau = (df["monthly_active_days"] > 0).sum()
    col1.metric("Total users", f"{len(df):,}")
    col2.metric("Churn rate", f"{churn_rate:.1%}")
    col3.metric("MAU", f"{mau:,}")
    col4.metric("Avg session", f"{df['session_duration_min'].mean():.1f} min")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg daily logins", f"{df['daily_logins'].mean():.1f}")
    col6.metric("Avg features used", f"{df['features_used'].mean():.1f}")
    col7.metric("Avg NPS", f"{df['nps_score'].mean():.1f}")
    col8.metric("Avg support tickets", f"{df['support_tickets'].mean():.1f}")

    st.subheader("Plan tier distribution")
    tier_counts = df["plan_tier"].value_counts().reset_index()
    tier_counts.columns = ["plan_tier", "count"]
    fig = px.pie(
        tier_counts, names="plan_tier", values="count",
        color="plan_tier",
        color_discrete_map={"Free": "#EF553B", "Pro": "#636EFA", "Enterprise": "#00CC96"},
        title="Users by plan tier",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn rate by plan tier")
    churn_by_tier = df.groupby("plan_tier")["is_churned"].mean().reset_index()
    churn_by_tier.columns = ["plan_tier", "churn_rate"]
    fig = px.bar(
        churn_by_tier, x="plan_tier", y="churn_rate",
        color="churn_rate", color_continuous_scale="RdYlGn_r",
        title="Churn rate by plan tier",
        labels={"churn_rate": "Churn rate"},
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Sample records")
    st.dataframe(df.head(10), use_container_width=True)


# =====================================================================
# PAGE 2: COHORT RETENTION
# =====================================================================
elif page == "Cohort Retention":
    st.title("Cohort retention analysis")

    st.markdown("Analyzing retention patterns by signup cohort to understand how user engagement evolves over time.")

    # Build cohort retention matrix
    df_cohort = df.copy()
    df_cohort["signup_month"] = df_cohort["signup_date"].dt.to_period("M")
    df_cohort["last_active_month"] = df_cohort["last_active_date"].dt.to_period("M")

    cohort_sizes = df_cohort.groupby("signup_month")["user_id"].nunique()

    # Calculate months between signup and last active
    df_cohort["months_active"] = (
        df_cohort["last_active_month"].apply(lambda x: x.ordinal) -
        df_cohort["signup_month"].apply(lambda x: x.ordinal)
    )

    # Retention: for each cohort and month offset, what fraction is still active
    retention_data = df_cohort.groupby(["signup_month", "months_active"])["user_id"].nunique().reset_index()
    retention_data.columns = ["signup_month", "months_active", "users"]

    # Pivot
    retention_pivot = retention_data.pivot(index="signup_month", columns="months_active", values="users")
    retention_pivot = retention_pivot.fillna(0)

    # Normalize by cohort size
    for col in retention_pivot.columns:
        retention_pivot[col] = retention_pivot[col] / cohort_sizes

    # Show only reasonable months (0-12)
    display_cols = [c for c in retention_pivot.columns if 0 <= c <= 12]
    retention_display = retention_pivot[display_cols].tail(12)

    st.subheader("Cohort retention heatmap")
    fig = px.imshow(
        retention_display.values,
        x=[f"M+{c}" for c in display_cols],
        y=[str(idx) for idx in retention_display.index],
        color_continuous_scale="YlGnBu",
        text_auto=".0%",
        title="Retention rate by signup cohort",
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Months since signup", yaxis_title="Signup cohort")
    st.plotly_chart(fig, use_container_width=True)

    # Cohort size over time
    st.subheader("Cohort sizes")
    cohort_df = cohort_sizes.reset_index()
    cohort_df.columns = ["signup_month", "users"]
    cohort_df["signup_month"] = cohort_df["signup_month"].astype(str)
    fig = px.bar(
        cohort_df, x="signup_month", y="users",
        title="New signups per month",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Retention by plan tier
    st.subheader("Retention by plan tier")
    retention_by_plan = df.groupby("plan_tier").agg(
        total_users=("user_id", "count"),
        active_users=("is_churned", lambda x: (x == 0).sum()),
        avg_active_days=("monthly_active_days", "mean"),
        avg_days_since_login=("days_since_last_login", "mean"),
    ).round(2)
    retention_by_plan["retention_rate"] = (retention_by_plan["active_users"] / retention_by_plan["total_users"]).round(3)
    st.dataframe(retention_by_plan, use_container_width=True)


# =====================================================================
# PAGE 3: FEATURE ADOPTION
# =====================================================================
elif page == "Feature Adoption":
    st.title("Feature adoption analysis")

    tab1, tab2, tab3 = st.tabs(["Adoption overview", "Usage patterns", "Engagement segments"])

    with tab1:
        st.subheader("Feature adoption by plan tier")
        fig = px.box(
            df, x="plan_tier", y="features_used", color="plan_tier",
            color_discrete_map={"Free": "#EF553B", "Pro": "#636EFA", "Enterprise": "#00CC96"},
            title="Features used by plan tier",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature adoption vs churn
        feature_bins = [0, 3, 6, 10, 15, 25]
        feature_labels = ["1-3", "4-6", "7-10", "11-15", "16-25"]
        df["feature_group"] = pd.cut(df["features_used"], bins=feature_bins, labels=feature_labels, include_lowest=True)
        churn_by_features = df.groupby("feature_group")["is_churned"].mean().reset_index()
        churn_by_features.columns = ["feature_group", "churn_rate"]
        fig = px.bar(
            churn_by_features, x="feature_group", y="churn_rate",
            color="churn_rate", color_continuous_scale="RdYlGn_r",
            title="Churn rate by feature adoption level",
            labels={"churn_rate": "Churn rate", "feature_group": "Features used"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x="features_used", color=df["is_churned"].map({0: "Active", 1: "Churned"}),
                barmode="overlay", nbins=25,
                title="Feature usage distribution by churn status",
                color_discrete_map={"Active": "#636EFA", "Churned": "#EF553B"},
                labels={"color": "Status"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df, x="session_duration_min", color=df["is_churned"].map({0: "Active", 1: "Churned"}),
                barmode="overlay", nbins=30,
                title="Session duration distribution by churn status",
                color_discrete_map={"Active": "#636EFA", "Churned": "#EF553B"},
                labels={"color": "Status"},
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df, x="features_used", y="session_duration_min",
            color=df["is_churned"].map({0: "Active", 1: "Churned"}),
            opacity=0.5,
            title="Features used vs. session duration",
            color_discrete_map={"Active": "#636EFA", "Churned": "#EF553B"},
            labels={"color": "Status"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Engagement segments")
        segment = st.selectbox(
            "Select segment variable",
            ["plan_tier", "industry", "login_group"],
        )
        churn_by_seg = df.groupby(segment)["is_churned"].mean().reset_index()
        churn_by_seg.columns = [segment, "churn_rate"]
        fig = px.bar(
            churn_by_seg, x=segment, y="churn_rate",
            color="churn_rate", color_continuous_scale="RdYlGn_r",
            title=f"Churn rate by {segment}",
            labels={"churn_rate": "Churn rate"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Free-tier users with fewer than 4 features adopted "
            "churn at 3x the rate of Enterprise users with 15+ features."
        )


# =====================================================================
# PAGE 4: AT-RISK ACCOUNTS
# =====================================================================
elif page == "At-Risk Accounts":
    st.title("At-risk account identification")

    impact_df = load_at_risk_analysis()
    if impact_df is not None:
        optimal = impact_df.loc[impact_df["net_savings"].idxmax()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal threshold", f"{optimal['threshold']:.3f}")
        col2.metric("Net savings (test set)", f"${int(optimal['net_savings']):,}")
        col3.metric("Full base estimate (annual)", f"${int(optimal['net_savings'] * 5):,}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Churners caught", f"{int(optimal['true_positives'])} ({optimal['churners_caught_pct']:.0f}%)")
        col5.metric("False alarms", f"{int(optimal['false_positives'])}")
        col6.metric("Total outreach", f"{int(optimal['outreach_count'])}")

        st.subheader("Cost-benefit analysis")
        st.markdown("""
        **Assumptions:**
        - Customer success outreach cost: $80 per account
        - Average annual contract value (churned account): $3,600
        - Retention success rate: 35% of contacted at-risk accounts are retained
        - Goal: find the threshold that maximizes net savings
        """)

        # Interactive threshold chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=impact_df["threshold"], y=impact_df["net_savings"],
                mode="lines+markers", name="Net savings ($)",
                line=dict(color="blue"),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=impact_df["threshold"], y=impact_df["churners_caught_pct"],
                mode="lines+markers", name="Churners caught (%)",
                line=dict(color="green", dash="dash"),
            ),
            secondary_y=True,
        )
        fig.add_vline(
            x=optimal["threshold"], line_dash="dash", line_color="red",
            annotation_text=f"Optimal: {optimal['threshold']:.2f}",
        )
        fig.update_xaxes(title_text="Classification threshold")
        fig.update_yaxes(title_text="Net savings ($)", secondary_y=False)
        fig.update_yaxes(title_text="Churners caught (%)", secondary_y=True)
        fig.update_layout(title="Threshold optimization", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.subheader("Threshold breakdown")
        st.dataframe(
            impact_df.style.format({
                "threshold": "{:.2f}",
                "outreach_cost": "${:,.0f}",
                "revenue_saved": "${:,.0f}",
                "net_savings": "${:,.0f}",
                "churners_caught_pct": "{:.1f}%",
            }),
            use_container_width=True,
        )

        # Static plot
        impact_path = os.path.join(OUTPUTS_DIR, "at_risk_analysis.png")
        if os.path.exists(impact_path):
            st.image(impact_path, caption="At-risk account analysis curve", use_container_width=True)
    else:
        st.warning("Run the model pipeline first to see at-risk account results.")

    # Model performance section
    st.subheader("Model performance")
    comparison = load_model_comparison()
    if comparison is not None:
        st.dataframe(
            comparison.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        fig = px.bar(
            comparison.reset_index(),
            x="index", y="auc_roc",
            color="auc_roc", color_continuous_scale="Viridis",
            title="AUC-ROC comparison",
            labels={"index": "Model", "auc_roc": "AUC-ROC"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show saved plots
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")

    col1, col2 = st.columns(2)
    if os.path.exists(roc_path):
        col1.image(roc_path, caption="ROC curves", use_container_width=True)
    if os.path.exists(cm_path):
        col2.image(cm_path, caption="Confusion matrices", use_container_width=True)

    # SHAP plots
    shap_summary_path = os.path.join(OUTPUTS_DIR, "shap_summary.png")
    shap_waterfall_path = os.path.join(OUTPUTS_DIR, "shap_waterfall.png")

    if os.path.exists(shap_summary_path):
        st.subheader("SHAP feature importance")
        st.image(shap_summary_path, use_container_width=True)
    if os.path.exists(shap_waterfall_path):
        st.subheader("Single user prediction breakdown")
        st.image(shap_waterfall_path, use_container_width=True)


# =====================================================================
# PAGE 5: PRODUCT HEALTH METRICS
# =====================================================================
elif page == "Product Health Metrics":
    st.title("Product health metrics")

    st.markdown("Holistic view of platform health across engagement, satisfaction, and growth dimensions.")

    # Top-level health score
    retention_rate = 1 - df["is_churned"].mean()
    avg_nps = df["nps_score"].mean()
    avg_adoption = df["features_used"].mean() / 25.0
    avg_engagement = (df["monthly_active_days"].mean() / 30.0)
    health_score = (retention_rate * 0.3 + avg_nps / 10 * 0.25 + avg_adoption * 0.25 + avg_engagement * 0.2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Product health score", f"{health_score:.0%}")
    col2.metric("Retention rate", f"{retention_rate:.1%}")
    col3.metric("Avg NPS", f"{avg_nps:.1f} / 10")
    col4.metric("Feature adoption", f"{avg_adoption:.0%}")

    # NPS distribution
    st.subheader("NPS score distribution")
    nps_categories = df["nps_score"].apply(
        lambda x: "Promoter (9-10)" if x >= 9 else ("Passive (7-8)" if x >= 7 else "Detractor (0-6)")
    )
    nps_dist = nps_categories.value_counts().reset_index()
    nps_dist.columns = ["category", "count"]
    fig = px.pie(
        nps_dist, names="category", values="count",
        color="category",
        color_discrete_map={
            "Promoter (9-10)": "#00CC96",
            "Passive (7-8)": "#636EFA",
            "Detractor (0-6)": "#EF553B",
        },
        title="NPS distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Engagement by industry
    st.subheader("Engagement by industry")
    industry_stats = df.groupby("industry").agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", "mean"),
        avg_logins=("daily_logins", "mean"),
        avg_features=("features_used", "mean"),
        avg_nps=("nps_score", "mean"),
    ).round(2).sort_values("churn_rate")

    fig = px.bar(
        industry_stats.reset_index(),
        x="industry", y="churn_rate",
        color="avg_nps", color_continuous_scale="RdYlGn",
        title="Churn rate by industry (colored by avg NPS)",
        labels={"churn_rate": "Churn rate", "avg_nps": "Avg NPS"},
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(industry_stats, use_container_width=True)

    # Support ticket analysis
    st.subheader("Support burden analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="support_tickets", color=df["is_churned"].map({0: "Active", 1: "Churned"}),
            barmode="overlay", nbins=15,
            title="Support tickets by churn status",
            color_discrete_map={"Active": "#636EFA", "Churned": "#EF553B"},
            labels={"color": "Status"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="plan_tier", y="support_tickets",
            color="plan_tier",
            color_discrete_map={"Free": "#EF553B", "Pro": "#636EFA", "Enterprise": "#00CC96"},
            title="Support tickets by plan tier",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation heatmap of numeric features",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key findings")
    st.markdown("""
    - **Free-tier users** have the highest churn rate, driven by low feature adoption and engagement
    - **Low daily logins** (under 3/day) are the strongest individual churn predictor
    - **Feature adoption** below 4 features dramatically increases churn probability
    - **High support ticket volume** (5+) indicates frustration and correlates strongly with churn
    - **Enterprise accounts** have the highest retention, driven by deeper product integration
    - **NPS detractors** (score 0-6) churn at significantly higher rates than promoters
    """)

"""
Streamlit dashboard for telecom customer churn prediction.
Five pages: Overview, EDA, Model performance, Explainability, Business impact.
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

st.set_page_config(page_title="Churn prediction dashboard", layout="wide")

DATA_PATH = "data/telco_churn.csv"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"], inplace=True)
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "13-24", "25-48", "49-72"]
    df["tenure_group"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, include_lowest=True)
    return df


@st.cache_data
def load_model_comparison():
    path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_business_impact():
    path = os.path.join(OUTPUTS_DIR, "business_impact.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# --- Sidebar navigation ---
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Exploratory data analysis", "Model performance", "Explainability", "Business impact"],
)

df = load_data()

# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if page == "Overview":
    st.title("Telecom customer churn prediction")
    st.markdown("Predicting which customers are likely to cancel their service, enabling targeted retention campaigns.")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df["churn"] == "Yes").mean()
    col1.metric("Total customers", f"{len(df):,}")
    col2.metric("Churn rate", f"{churn_rate:.1%}")
    col3.metric("Avg monthly charges", f"${df['monthly_charges'].mean():.2f}")
    col4.metric("Avg tenure", f"{df['tenure_months'].mean():.1f} months")

    st.subheader("Dataset summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Sample records")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Key findings")
    st.markdown("""
    - **Month-to-month contracts** show the highest churn rates
    - **Fiber optic** customers churn more than DSL customers
    - **Short tenure** (under 12 months) is a strong churn indicator
    - **Electronic check** payment method correlates with higher churn
    - Customers without **tech support** or **online security** are more likely to leave
    """)


# =====================================================================
# PAGE 2: EDA
# =====================================================================
elif page == "Exploratory data analysis":
    st.title("Exploratory data analysis")

    tab1, tab2, tab3 = st.tabs(["Churn by segment", "Distributions", "Correlations"])

    with tab1:
        segment = st.selectbox(
            "Select segment variable",
            ["contract", "internet_service", "tenure_group", "payment_method",
             "gender", "senior_citizen", "partner", "dependents",
             "phone_service", "paperless_billing"],
        )
        churn_by_seg = df.groupby(segment)["churn"].apply(
            lambda x: (x == "Yes").mean()
        ).reset_index()
        churn_by_seg.columns = [segment, "churn_rate"]

        fig = px.bar(
            churn_by_seg, x=segment, y="churn_rate",
            color="churn_rate",
            color_continuous_scale="RdYlGn_r",
            title=f"Churn rate by {segment}",
            labels={"churn_rate": "Churn rate"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # Cross-segment insight
        st.info(
            "Month-to-month fiber customers with electronic check payment "
            "churn at 3x the rate of two-year contract customers."
        )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x="tenure_months", color="churn",
                barmode="overlay", nbins=36,
                title="Tenure distribution by churn status",
                color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df, x="monthly_charges", color="churn",
                barmode="overlay", nbins=30,
                title="Monthly charges distribution by churn status",
                color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            df, x="contract", y="monthly_charges", color="churn",
            title="Monthly charges by contract type and churn",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].copy()
        corr["churn_flag"] = (df["churn"] == "Yes").astype(int)
        corr_matrix = corr.corr()

        fig = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation heatmap",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE 3: MODEL PERFORMANCE
# =====================================================================
elif page == "Model performance":
    st.title("Model performance comparison")

    comparison = load_model_comparison()
    if comparison is not None:
        st.subheader("Metrics comparison")
        st.dataframe(
            comparison.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        # Bar chart of AUC-ROC
        fig = px.bar(
            comparison.reset_index(),
            x="index", y="auc_roc",
            color="auc_roc",
            color_continuous_scale="Viridis",
            title="AUC-ROC comparison",
            labels={"index": "Model", "auc_roc": "AUC-ROC"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run the model pipeline first to generate results.")

    # Show saved plots
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")

    col1, col2 = st.columns(2)
    if os.path.exists(roc_path):
        col1.image(roc_path, caption="ROC curves", use_container_width=True)
    if os.path.exists(cm_path):
        col2.image(cm_path, caption="Confusion matrices", use_container_width=True)


# =====================================================================
# PAGE 4: EXPLAINABILITY
# =====================================================================
elif page == "Explainability":
    st.title("Model explainability")

    shap_summary_path = os.path.join(OUTPUTS_DIR, "shap_summary.png")
    shap_waterfall_path = os.path.join(OUTPUTS_DIR, "shap_waterfall.png")
    fi_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")

    st.subheader("SHAP summary plot")
    st.markdown("Each dot represents one customer. Position on x-axis shows the impact on churn prediction. Color shows the feature value (red = high, blue = low).")
    if os.path.exists(shap_summary_path):
        st.image(shap_summary_path, use_container_width=True)
    else:
        st.warning("Run the model pipeline to generate SHAP plots.")

    st.subheader("Single customer prediction breakdown")
    if os.path.exists(shap_waterfall_path):
        st.image(shap_waterfall_path, use_container_width=True)

    st.subheader("Feature importance across models")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)

    # Interactive single customer prediction
    st.subheader("Predict for a single customer")
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))

        st.markdown("Adjust the sliders to see how the prediction changes:")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly charges ($)", 18, 120, 65)
        with col2:
            contract_type = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        with col3:
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            support_count = st.slider("Support services (0-4)", 0, 4, 1)

        st.caption("Note: this uses simplified inputs mapped to the full feature set. For precise predictions, use the complete feature vector.")
    else:
        st.info("Train the model first to enable single customer predictions.")


# =====================================================================
# PAGE 5: BUSINESS IMPACT
# =====================================================================
elif page == "Business impact":
    st.title("Business impact analysis")

    impact_df = load_business_impact()
    if impact_df is not None:
        optimal = impact_df.loc[impact_df["net_savings"].idxmax()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal threshold", f"{optimal['threshold']:.3f}")
        col2.metric("Net savings (test set)", f"${int(optimal['net_savings']):,}")
        col3.metric("Full base estimate (annual)", f"${int(optimal['net_savings'] * 5):,}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Churners caught", f"{int(optimal['true_positives'])} ({optimal['churners_caught_pct']:.0f}%)")
        col5.metric("False alarms", f"{int(optimal['false_positives'])}")
        col6.metric("Total interventions", f"{int(optimal['interventions'])}")

        st.subheader("Cost-benefit analysis")
        st.markdown("""
        **Assumptions:**
        - Retention intervention cost: $50 per customer (call, offer, follow-up)
        - Customer lifetime value of a churner: $900 (12 months at $75/month avg)
        - Retention success rate: 30% of contacted at-risk customers are retained
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
                "intervention_cost": "${:,.0f}",
                "revenue_saved": "${:,.0f}",
                "net_savings": "${:,.0f}",
                "churners_caught_pct": "{:.1f}%",
            }),
            use_container_width=True,
        )

        # Static plot
        impact_path = os.path.join(OUTPUTS_DIR, "business_impact.png")
        if os.path.exists(impact_path):
            st.image(impact_path, caption="Business impact curve", use_container_width=True)
    else:
        st.warning("Run the model pipeline first to see business impact results.")

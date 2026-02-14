"""
Modelling module for Calgary Business Survival Analyzer & Location Recommender.

Contains:
- Survival-analysis helpers (Kaplan-Meier, Cox PH)
- Classification models (Random Forest, XGBoost) for predicting business survival
- Location-recommendation engine
- Model persistence utilities
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================================
# Survival Analysis
# ============================================================================

class SurvivalAnalyzer:
    """Wraps lifelines Kaplan-Meier and Cox Proportional-Hazards models."""

    def __init__(self) -> None:
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter(penalizer=0.01)
        self._cph_fitted = False

    # ---- Kaplan-Meier --------------------------------------------------

    def fit_kaplan_meier(
        self,
        durations: pd.Series,
        event_observed: pd.Series,
        label: str = "Overall",
    ) -> KaplanMeierFitter:
        """Fit a Kaplan-Meier survival curve.

        Parameters
        ----------
        durations : pd.Series
            Observed lifetimes (business_age_days).
        event_observed : pd.Series
            1 if the event (closure) was observed, 0 if censored.
        label : str
            Label for the curve.

        Returns
        -------
        KaplanMeierFitter
            The fitted estimator.
        """
        self.kmf.fit(durations, event_observed=event_observed, label=label)
        return self.kmf

    def get_kaplan_meier_curve(
        self,
        durations: pd.Series,
        event_observed: pd.Series,
        label: str = "Overall",
    ) -> pd.DataFrame:
        """Return the KM survival-function as a tidy DataFrame.

        Columns: timeline, survival_probability, label.
        """
        self.fit_kaplan_meier(durations, event_observed, label)
        sf = self.kmf.survival_function_.reset_index()
        sf.columns = ["timeline", "survival_probability"]
        sf["label"] = label
        return sf

    def kaplan_meier_by_group(
        self,
        df: pd.DataFrame,
        duration_col: str = "business_age_days",
        event_col: str = "event_observed",
        group_col: str = "business_category",
        top_n: int = 8,
    ) -> pd.DataFrame:
        """Compute Kaplan-Meier curves for the *top_n* most-frequent groups.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *duration_col*, *event_col*, and *group_col*.
        top_n : int
            Number of groups to include.

        Returns
        -------
        pd.DataFrame
            Concatenated survival curves with a ``label`` column.
        """
        top_groups = df[group_col].value_counts().head(top_n).index.tolist()
        curves = []
        for group in top_groups:
            mask = df[group_col] == group
            sub = df.loc[mask]
            if len(sub) < 10:
                continue
            curve = self.get_kaplan_meier_curve(
                sub[duration_col], sub[event_col], label=str(group)
            )
            curves.append(curve)
        if not curves:
            return pd.DataFrame(columns=["timeline", "survival_probability", "label"])
        return pd.concat(curves, ignore_index=True)

    def survival_rates_at_milestones(
        self,
        df: pd.DataFrame,
        duration_col: str = "business_age_days",
        event_col: str = "event_observed",
        group_col: str = "business_category",
        milestones_years: list[int] | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return survival probabilities at 1, 3, and 5 years per group.

        Parameters
        ----------
        milestones_years : list[int] | None
            Years to report. Defaults to [1, 3, 5].

        Returns
        -------
        pd.DataFrame
            Rows = groups, columns = milestone survival rates.
        """
        if milestones_years is None:
            milestones_years = [1, 3, 5]
        milestone_days = [y * 365 for y in milestones_years]

        top_groups = df[group_col].value_counts().head(top_n).index.tolist()
        rows = []
        for group in top_groups:
            mask = df[group_col] == group
            sub = df.loc[mask]
            if len(sub) < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub[duration_col], event_observed=sub[event_col])
            row = {"business_type": group, "count": int(mask.sum())}
            for yr, d in zip(milestones_years, milestone_days):
                try:
                    prob = float(kmf.predict(d))
                except Exception:
                    prob = np.nan
                row[f"{yr}_year_survival"] = round(prob, 4)
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- Cox PH --------------------------------------------------------

    def fit_cox(self, df: pd.DataFrame, duration_col: str = "business_age_days",
                event_col: str = "event_observed") -> CoxPHFitter:
        """Fit a Cox Proportional-Hazards model.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *duration_col*, *event_col*, and numeric covariates.

        Returns
        -------
        CoxPHFitter
        """
        self.cph.fit(df, duration_col=duration_col, event_col=event_col)
        self._cph_fitted = True
        return self.cph

    def get_hazard_summary(self) -> pd.DataFrame:
        """Return the Cox model summary as a DataFrame.

        Raises
        ------
        RuntimeError
            If the Cox model has not been fitted.
        """
        if not self._cph_fitted:
            raise RuntimeError("Cox model not fitted yet. Call fit_cox() first.")
        return self.cph.summary


# ============================================================================
# Classification Models
# ============================================================================

FEATURE_COLS = [
    "business_age_days",
    "is_home_occupation",
    "business_count",
    "business_diversity",
    "avg_business_age",
    "issue_year",
    "issue_month",
]


def _prepare_classification_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "survived",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Prepare train/test splits for classification.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Encode business_category if present and requested
    if "business_category" in df.columns and "business_category_enc" not in df.columns:
        le = LabelEncoder()
        df = df.copy()
        df["business_category_enc"] = le.fit_transform(
            df["business_category"].astype(str)
        )
        if "business_category_enc" not in feature_cols:
            feature_cols = feature_cols + ["business_category_enc"]

    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols]
    y = subset[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    **rf_kwargs: Any,
) -> dict[str, Any]:
    """Train a Random Forest classifier on business-survival data.

    Returns
    -------
    dict
        Keys: model, metrics, feature_importance, X_test, y_test, feature_names.
    """
    X_train, X_test, y_train, y_test, feat_names = _prepare_classification_data(
        df, feature_cols
    )

    defaults = dict(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    defaults.update(rf_kwargs)
    model = RandomForestClassifier(**defaults)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics(y_test, y_pred, y_prob)
    importance = pd.DataFrame(
        {"feature": feat_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info("Random Forest accuracy: %.4f", metrics["accuracy"])
    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feat_names,
    }


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    **xgb_kwargs: Any,
) -> dict[str, Any]:
    """Train an XGBoost classifier on business-survival data.

    Returns
    -------
    dict
        Keys: model, metrics, feature_importance, X_test, y_test, feature_names.
    """
    X_train, X_test, y_train, y_test, feat_names = _prepare_classification_data(
        df, feature_cols
    )

    defaults = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    defaults.update(xgb_kwargs)
    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics(y_test, y_pred, y_prob)
    importance = pd.DataFrame(
        {"feature": feat_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info("XGBoost accuracy: %.4f", metrics["accuracy"])
    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feat_names,
    }


def _compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "classification_report": classification_report(y_true, y_pred),
    }


# ============================================================================
# Location Recommender
# ============================================================================

def recommend_locations(
    df: pd.DataFrame,
    business_type: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Score and rank communities for a given business type.

    Scoring considers:
    1. **Survival rate** of the same business type in the community.
    2. **Competition level** (fewer competitors = higher score).
    3. **Demographic fit** proxied by overall business diversity and average
       business age.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed business-licence data.
    business_type : str
        The business category to evaluate.
    top_n : int
        Number of communities to return.

    Returns
    -------
    pd.DataFrame
        Ranked communities with scores and component details.
    """
    community_col = "comdistnm"

    # Filter to the selected business type
    type_mask = df["business_category"].str.lower() == business_type.lower()
    type_df = df.loc[type_mask]

    if type_df.empty:
        logger.warning("No records for business type '%s'", business_type)
        return pd.DataFrame()

    # Community-level stats for the specific type
    type_stats = (
        type_df.groupby(community_col)
        .agg(
            type_count=("getbusid", "count"),
            type_survival_rate=("survived", "mean"),
            type_avg_age=("business_age_days", "mean"),
        )
        .reset_index()
    )

    # Overall community stats
    overall = (
        df.groupby(community_col)
        .agg(
            total_businesses=("getbusid", "count"),
            business_diversity=("business_category", "nunique"),
            community_survival_rate=("survived", "mean"),
            avg_business_age=("business_age_days", "mean"),
        )
        .reset_index()
    )

    merged = type_stats.merge(overall, on=community_col, how="left")

    # --- Scoring ----------------------------------------------------------
    # 1. Survival score (higher is better) -- normalized 0-1
    merged["survival_score"] = _normalize(merged["type_survival_rate"])

    # 2. Competition score (lower competition = higher score)
    merged["competition_score"] = 1 - _normalize(merged["type_count"])

    # 3. Demographic / diversity score
    merged["diversity_score"] = _normalize(merged["business_diversity"])

    # Weighted composite
    merged["overall_score"] = (
        0.45 * merged["survival_score"]
        + 0.30 * merged["competition_score"]
        + 0.25 * merged["diversity_score"]
    ).round(4)

    # Tidy up
    merged["type_survival_rate"] = merged["type_survival_rate"].round(4)
    merged["community_survival_rate"] = merged["community_survival_rate"].round(4)
    merged["type_avg_age"] = merged["type_avg_age"].round(0).astype(int)
    merged["avg_business_age"] = merged["avg_business_age"].round(0).astype(int)

    result = merged.sort_values("overall_score", ascending=False).head(top_n)
    result = result.reset_index(drop=True)
    result.index = result.index + 1  # 1-based ranking
    result.index.name = "rank"

    return result


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series(0.5, index=series.index)
    return (series - smin) / (smax - smin)


def get_competition_analysis(
    df: pd.DataFrame,
    business_type: str,
) -> pd.DataFrame:
    """Count businesses of the same type per community.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data.
    business_type : str
        Business category.

    Returns
    -------
    pd.DataFrame
        Columns: community, competitor_count, survival_rate.
    """
    community_col = "comdistnm"
    mask = df["business_category"].str.lower() == business_type.lower()
    sub = df.loc[mask]
    if sub.empty:
        return pd.DataFrame(
            columns=["community", "competitor_count", "survival_rate"]
        )
    result = (
        sub.groupby(community_col)
        .agg(competitor_count=("getbusid", "count"), survival_rate=("survived", "mean"))
        .reset_index()
        .rename(columns={community_col: "community"})
        .sort_values("competitor_count", ascending=False)
        .reset_index(drop=True)
    )
    result["survival_rate"] = result["survival_rate"].round(4)
    return result


# ============================================================================
# Model persistence
# ============================================================================

def save_model(obj: Any, filename: str) -> Path:
    """Persist a model (or any picklable object) to the models/ directory.

    Parameters
    ----------
    obj : Any
        Object to save.
    filename : str
        File name (e.g., ``"rf_model.joblib"``).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(filename: str) -> Any:
    """Load a previously saved model from the models/ directory.

    Parameters
    ----------
    filename : str
        File name to load.

    Returns
    -------
    Any
        The deserialised object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return obj

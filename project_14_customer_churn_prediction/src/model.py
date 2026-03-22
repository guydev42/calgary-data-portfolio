"""
Train, evaluate, and explain churn prediction models.
Includes: Logistic Regression, Random Forest, XGBoost, LightGBM.
Generates SHAP explanations and business impact analysis.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _get_models_and_grids():
    """Return model instances and small hyperparameter grids."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "params": {"C": [0.01, 0.1, 1.0, 10.0]},
            "needs_scaling": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
            },
            "needs_scaling": False,
        },
    }

    try:
        from xgboost import XGBClassifier
        # Compute scale_pos_weight for imbalanced data
        models["XGBoost"] = {
            "model": XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
            },
            "needs_scaling": False,
        }
    except ImportError:
        print("XGBoost not installed, skipping.")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = {
            "model": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7, -1],
                "learning_rate": [0.05, 0.1],
            },
            "needs_scaling": False,
        }
    except ImportError:
        print("LightGBM not installed, skipping.")

    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all models with GridSearchCV, evaluate, generate SHAP plots,
    and compute business impact. Saves the best model to disk.
    """
    _ensure_dirs()

    # Scale data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = _get_models_and_grids()
    results = {}
    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    for name, config in models_config.items():
        print(f"\n--- {name} ---")

        Xtr = X_train_scaled if config["needs_scaling"] else X_train
        Xte = X_test_scaled if config["needs_scaling"] else X_test

        grid = GridSearchCV(
            config["model"], config["params"],
            cv=5, scoring="roc_auc", n_jobs=-1, refit=True,
        )
        grid.fit(Xtr, y_train)

        best_model = grid.best_estimator_
        trained_models[name] = {
            "model": best_model,
            "needs_scaling": config["needs_scaling"],
        }

        y_pred = best_model.predict(Xte)
        y_prob = best_model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc_roc": auc, "confusion_matrix": cm,
            "y_prob": y_prob, "best_params": grid.best_params_,
        }

        print(f"  Best params: {grid.best_params_}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Confusion matrix:")
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # --- Model comparison table ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k not in ("confusion_matrix", "y_prob", "best_params")}
        for name, r in results.items()
    }).T.round(4)
    print(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))

    # --- Find and save best model ---
    best_name = max(results, key=lambda n: results[n]["auc_roc"])
    best_auc = results[best_name]["auc_roc"]
    print(f"\nBest model: {best_name} (AUC-ROC = {best_auc:.4f})")

    best_info = trained_models[best_name]
    joblib.dump(best_info["model"], os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))
    print(f"Saved best model to {MODELS_DIR}/best_model.joblib")

    # --- ROC curves ---
    _plot_roc_curves(results, y_test)

    # --- Confusion matrices ---
    _plot_confusion_matrices(results)

    # --- Feature importance comparison ---
    _plot_feature_importance(trained_models, feature_names, X_train, X_train_scaled)

    # --- SHAP explainability ---
    _generate_shap(trained_models, feature_names, X_test, X_test_scaled, best_name)

    # --- Business impact analysis ---
    best_probs = results[best_name]["y_prob"]
    _business_impact(best_probs, y_test, best_name)

    return results


def _plot_roc_curves(results, y_test):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - model comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved ROC curves plot.")


def _plot_confusion_matrices(results):
    """Plot confusion matrices side by side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(
            r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["No churn", "Churn"],
            yticklabels=["No churn", "Churn"],
            ax=ax,
        )
        ax.set_title(f"{name}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    fig.suptitle("Confusion matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion matrices plot.")


def _plot_feature_importance(trained_models, feature_names, X_train, X_train_scaled):
    """Plot feature importance from tree-based models."""
    tree_models = {n: v for n, v in trained_models.items() if not v["needs_scaling"]}
    if not tree_models:
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 8))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, tree_models.items()):
        model = info["model"]
        importances = model.feature_importances_
        idx = np.argsort(importances)[-15:]
        ax.barh(range(len(idx)), importances[idx], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.set_title(f"{name} - top 15 features")
        ax.set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("Saved feature importance plot.")


def _generate_shap(trained_models, feature_names, X_test, X_test_scaled, best_name):
    """Generate SHAP summary and waterfall plots for the best model."""
    print("\nGenerating SHAP explanations...")

    info = trained_models[best_name]
    model = info["model"]
    X = X_test_scaled if info["needs_scaling"] else X_test

    # Use a subsample for speed
    sample_size = min(500, X.shape[0])
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    X_df = pd.DataFrame(X_sample, columns=feature_names)

    if info["needs_scaling"]:
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # For tree models, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=20)
    plt.title("SHAP feature impact on churn prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved SHAP summary plot.")

    # Waterfall plot for a single customer (first predicted churner)
    try:
        probs = model.predict_proba(X_sample)[:, 1]
        churn_idx = np.where(probs > 0.5)[0]
        if len(churn_idx) > 0:
            single_idx = churn_idx[0]
        else:
            single_idx = 0

        if isinstance(shap_values, list):
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            base_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

        explanation = shap.Explanation(
            values=shap_vals[single_idx],
            base_values=base_val,
            data=X_sample[single_idx],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False, max_display=15)
        plt.title("SHAP waterfall - single customer prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        print("Saved SHAP waterfall plot.")
    except Exception as e:
        print(f"Waterfall plot skipped: {e}")

    # Save SHAP values for the app
    np.save(os.path.join(OUTPUTS_DIR, "shap_values.npy"), shap_vals)
    np.save(os.path.join(OUTPUTS_DIR, "shap_sample.npy"), X_sample)
    print("Saved SHAP values to disk.")


def _business_impact(y_prob, y_test, model_name):
    """
    Cost-benefit analysis for churn intervention.

    Assumptions:
    - Retention intervention cost: $50 per customer contacted
    - Average customer lifetime value (CLV) of a churner: $900 (12 months * $75/month)
    - Retention success rate: 30% of contacted churners are successfully retained
    - If we catch a true churner AND retain them (30% chance), we save $900
    - If we intervene on a non-churner (false positive), we waste $50
    - Missed churners cost us the full CLV of $900
    """
    print("\n" + "=" * 70)
    print("BUSINESS IMPACT ANALYSIS")
    print("=" * 70)

    intervention_cost = 50       # cost to contact/offer retention deal
    clv_churner = 900            # 12 months * $75 avg revenue
    retention_rate = 0.30        # 30% of contacted churners stay

    thresholds = np.arange(0.05, 0.91, 0.025)
    impact_records = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        total_interventions = tp + fp
        cost_of_interventions = total_interventions * intervention_cost
        retained_churners = tp * retention_rate
        revenue_saved = retained_churners * clv_churner
        net_savings = revenue_saved - cost_of_interventions
        churners_caught_pct = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

        impact_records.append({
            "threshold": round(t, 3),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "interventions": total_interventions,
            "intervention_cost": cost_of_interventions,
            "revenue_saved": round(revenue_saved, 0),
            "net_savings": round(net_savings, 0),
            "churners_caught_pct": round(churners_caught_pct, 1),
        })

    impact_df = pd.DataFrame(impact_records)

    # Find optimal threshold
    optimal_idx = impact_df["net_savings"].idxmax()
    optimal = impact_df.loc[optimal_idx]

    print(f"\nModel: {model_name}")
    print(f"Intervention cost per customer:  ${intervention_cost}")
    print(f"Customer lifetime value (churner): ${clv_churner}")
    print(f"Retention success rate:           {retention_rate:.0%}")
    print(f"\nOptimal threshold: {optimal['threshold']:.3f}")
    print(f"  Churners caught:    {int(optimal['true_positives'])} / {int(optimal['true_positives'] + optimal['false_negatives'])} ({optimal['churners_caught_pct']:.1f}%)")
    print(f"  False positives:    {int(optimal['false_positives'])}")
    print(f"  Total interventions: {int(optimal['interventions'])}")
    print(f"  Intervention cost:  ${int(optimal['intervention_cost']):,}")
    print(f"  Revenue saved:      ${int(optimal['revenue_saved']):,}")
    print(f"  NET SAVINGS:        ${int(optimal['net_savings']):,}")

    # Scale to full customer base (test set is 20% of 5000)
    scale_factor = 5
    annual_savings = int(optimal["net_savings"]) * scale_factor
    print(f"  Full base estimate: ${annual_savings:,}/year (scaled from test set)")

    # Save impact data
    impact_df.to_csv(os.path.join(OUTPUTS_DIR, "business_impact.csv"), index=False)

    # Plot cost-benefit curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(impact_df["threshold"], impact_df["net_savings"], "b-o", markersize=4, label="Net savings ($)")
    ax1.axvline(x=optimal["threshold"], color="red", linestyle="--", alpha=0.7, label=f"Optimal threshold ({optimal['threshold']:.3f})")
    ax1.set_xlabel("Classification threshold")
    ax1.set_ylabel("Net savings ($)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(impact_df["threshold"], impact_df["churners_caught_pct"], "g--s", markersize=4, alpha=0.7, label="Churners caught (%)")
    ax2.set_ylabel("Churners caught (%)", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.legend(loc="upper right")

    plt.title(f"Business impact - {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "business_impact.png"), dpi=150)
    plt.close(fig)
    print("Saved business impact plot.")

    return impact_df, optimal


if __name__ == "__main__":
    from data_loader import load_and_prepare
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()
    train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

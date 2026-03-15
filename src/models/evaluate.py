"""
evaluate.py — Model Evaluation & Reporting (Stage 7)
======================================================
Full evaluation module for B2B SaaS churn prediction. Evaluates all four
trained models (Dummy, Logistic Regression, Random Forest, Gradient Boosting).

Outputs:
    1. Summary comparison DataFrame (sorted by Recall descending)
    2. Confusion matrix heatmaps with TN/FP/FN/TP labels
    3. ROC curves — all models overlaid, AUC in legend
    4. Precision-Recall curves — all models overlaid
    5. Feature importances (top 15) for RF and GB with interpretation
    6. Class imbalance handling report (balanced vs. unbalanced)
    7. Business interpretation markdown report

All plots saved to reports/figures/.
No SHAP — only sklearn feature_importances_.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Logging & Style
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42


# ===========================================================================
# 1. Core Metrics
# ===========================================================================
def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "model",
) -> Dict[str, float]:
    """
    Compute all key classification metrics for the churn (positive) class.

    Returns dict: accuracy, precision_churn, recall_churn, f1_churn,
                  roc_auc, brier_score.
    """
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_churn": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_churn": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_churn": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["brier_score"] = brier_score_loss(y_true, y_proba)
        except ValueError:
            metrics["brier_score"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["brier_score"] = float("nan")

    return metrics


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all models and return a summary comparison DataFrame.
    Sorted by Recall descending (catching churners is the business priority).
    """
    rows = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba, model_name=name)
        metrics["model"] = name
        rows.append(metrics)

    comparison = pd.DataFrame(rows)
    # Reorder columns: Model first, then the required metrics
    col_order = ["model", "accuracy", "precision_churn", "recall_churn", "f1_churn", "roc_auc"]
    extra = [c for c in comparison.columns if c not in col_order]
    comparison = comparison[col_order + extra]
    comparison = comparison.sort_values("recall_churn", ascending=False).reset_index(drop=True)

    return comparison


# ===========================================================================
# 2. Classification Report (per model)
# ===========================================================================
def print_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "model",
    output_dir: Optional[str] = None,
) -> str:
    """Print and optionally save sklearn classification report."""
    report = classification_report(
        y_true, y_pred, target_names=["Not Churned", "Churned"]
    )
    logger.info("\n=== %s Classification Report ===\n%s", model_name, report)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{model_name}_classification_report.txt"
        filepath.write_text(f"=== {model_name} ===\n{report}")
        logger.info("Saved report to %s", filepath)

    return report


# ===========================================================================
# 3. Confusion Matrix — with TN/FP/FN/TP labels
# ===========================================================================
def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "model",
    save: bool = True,
) -> None:
    """
    Plot confusion matrix as a seaborn heatmap with TN/FP/FN/TP labels
    in each cell alongside the count.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Create annotation labels with both count and label
    labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png")
        logger.info("Saved confusion matrix plot for %s", model_name)
    plt.close(fig)


# ===========================================================================
# 4. ROC Curve — all models on one plot, AUC in legend
# ===========================================================================
def plot_roc_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot overlaid ROC curves for all models with AUC in legend."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("husl", len(trained_models))

    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc_val:.3f})",
            )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curves_comparison.png")
        logger.info("Saved ROC curves comparison plot")
    plt.close(fig)


# ===========================================================================
# 5. Precision-Recall Curve — all models on one plot
# ===========================================================================
def plot_precision_recall_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """
    Plot overlaid Precision-Recall curves for all models.
    More informative than ROC under class imbalance.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("husl", len(trained_models))

    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            pr_auc_val = auc(recall_vals, precision_vals)
            ax.plot(
                recall_vals, precision_vals, color=color, lw=2,
                label=f"{name} (PR-AUC = {pr_auc_val:.3f})",
            )

    # Baseline = churn prevalence
    baseline = y_test.mean()
    ax.axhline(
        y=baseline, color="gray", linestyle="--", lw=1,
        label=f"Baseline (prevalence = {baseline:.3f})",
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves — All Models\n(more informative under class imbalance)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "precision_recall_curves.png")
        logger.info("Saved Precision-Recall curves plot")
    plt.close(fig)


# ===========================================================================
# 6. Feature Importances — Top 15 for RF and GB
# ===========================================================================
def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "model",
    top_n: int = 15,
    save: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Plot top-N feature importance bar chart for tree-based models.
    Returns DataFrame of importances sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("Model '%s' has no feature_importances_ attribute.", model_name)
        return None

    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(data=fi_df, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_xlabel("Feature Importance (Gini / Mean Decrease Impurity)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"Top {top_n} Features — {model_name}",
        fontsize=14, fontweight="bold",
    )

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{model_name}_feature_importance.png")
        logger.info("Saved feature importance plot for %s", model_name)
    plt.close(fig)

    return fi_df


def interpret_feature_importances(
    fi_rf: Optional[pd.DataFrame],
    fi_gb: Optional[pd.DataFrame],
) -> str:
    """
    Interpret top features and identify SaaS churn signals.
    Returns a markdown-formatted interpretation string.
    """
    lines = [
        "### Feature Importance Interpretation",
        "",
        "Top features driving churn predictions (from tree-based models):",
        "",
    ]

    # Combine importances from both models
    all_features = set()
    if fi_rf is not None:
        all_features.update(fi_rf["feature"].tolist()[:5])
    if fi_gb is not None:
        all_features.update(fi_gb["feature"].tolist()[:5])

    # Map feature names to business interpretations
    interpretations = {
        "usage_intensity": "Low events per month = disengaged customer, strong churn signal",
        "tenure_months": "Short tenure = higher churn risk (haven't seen full product value)",
        "active_day_ratio": "Low consistency of usage (sporadic engagement) = at-risk",
        "avg_errors": "High error rates degrade user experience and drive churn",
        "avg_duration_secs": "Session duration patterns reveal engagement depth",
        "days_since_last_activity": "Recent inactivity is a leading churn indicator",
        "mrr_amount": "Revenue level -- different retention tactics needed by segment",
        "ticket_count": "High support volume may indicate frustration or product issues",
        "avg_satisfaction": "Low satisfaction scores correlate with intent to churn",
        "min_satisfaction": "Even one bad support experience can trigger churn",
        "avg_resolution_hours": "Slow resolution times erode customer confidence",
        "total_events": "Overall engagement volume — low = disengaged",
        "active_days": "Fewer active days = less product stickiness",
        "seats": "Seat count — larger accounts may have different churn dynamics",
        "avg_daily_usage": "Low daily usage = not embedding product in workflow",
        "escalation_count": "Escalations signal serious unresolved issues",
        "total_errors": "High error rates degrade user experience",
        "unique_features_used": "Low feature adoption = not seeing full product value",
        "beta_feature_usage": "Beta feature adoption shows engagement with innovation",
        "avg_first_response_min": "Slow first response to tickets drives dissatisfaction",
        "near_renewal": "Accounts near renewal are at a decision point",
    }

    if fi_rf is not None:
        lines.append("**Random Forest — Top 5:**")
        for _, row in fi_rf.head(5).iterrows():
            feat = row["feature"]
            # Clean up feature names from ColumnTransformer (e.g., "num__usage_intensity")
            clean_name = feat.split("__")[-1] if "__" in feat else feat
            interp = interpretations.get(clean_name, "Contributes to churn prediction")
            lines.append(f"  - `{feat}` ({row['importance']:.4f}): {interp}")
        lines.append("")

    if fi_gb is not None:
        lines.append("**Gradient Boosting — Top 5:**")
        for _, row in fi_gb.head(5).iterrows():
            feat = row["feature"]
            clean_name = feat.split("__")[-1] if "__" in feat else feat
            interp = interpretations.get(clean_name, "Contributes to churn prediction")
            lines.append(f"  - `{feat}` ({row['importance']:.4f}): {interp}")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# 7. Class Imbalance Handling Report
# ===========================================================================
def run_class_imbalance_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, str]:
    """
    Compare models WITH and WITHOUT class_weight='balanced'.
    Shows how Recall and F1 for the churn class change.

    Returns (comparison_df, markdown_report).
    """
    logger.info("=" * 60)
    logger.info("CLASS IMBALANCE HANDLING REPORT")
    logger.info("=" * 60)

    models_config = {
        "LogReg (no balancing)": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs",
        ),
        "LogReg (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, solver="lbfgs",
        ),
        "RF (no balancing)": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "RF (balanced)": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GB (no balancing)": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
        ),
        "GB (balanced — sample_weight)": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
        ),
    }

    rows = []
    for name, model in models_config.items():
        # GradientBoostingClassifier doesn't support class_weight;
        # use sample_weight instead
        if "sample_weight" in name.lower():
            # Compute sample weights proportional to class imbalance
            n_samples = len(y_train)
            n_pos = y_train.sum()
            n_neg = n_samples - n_pos
            weight_pos = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
            weight_neg = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
            sample_weights = y_train.map({0: weight_neg, 1: weight_pos}).values
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_proba, model_name=name)
        metrics["model"] = name
        rows.append(metrics)

    comparison = pd.DataFrame(rows)
    col_order = ["model", "accuracy", "precision_churn", "recall_churn", "f1_churn", "roc_auc"]
    comparison = comparison[[c for c in col_order if c in comparison.columns]]
    comparison = comparison.sort_values("recall_churn", ascending=False).reset_index(drop=True)

    # Build markdown
    md_lines = [
        "### Class Imbalance Handling Report",
        "",
        "Comparison of models **with** and **without** `class_weight='balanced'`:",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "**Key Findings:**",
    ]

    # Find the effect of balancing
    for base in ["LogReg", "RF", "GB"]:
        balanced_row = comparison[comparison["model"].str.contains(f"{base}.*balanc", case=False)]
        unbalanced_row = comparison[comparison["model"].str.contains(f"{base}.*no balanc", case=False)]
        if not balanced_row.empty and not unbalanced_row.empty:
            recall_diff = float(balanced_row["recall_churn"].iloc[0]) - float(unbalanced_row["recall_churn"].iloc[0])
            f1_diff = float(balanced_row["f1_churn"].iloc[0]) - float(unbalanced_row["f1_churn"].iloc[0])
            md_lines.append(
                f"- **{base}**: Balanced weights -- Recall delta={recall_diff:+.3f}, F1 delta={f1_diff:+.3f}"
            )

    md_lines.extend([
        "",
        "> Class balancing trades accuracy for recall -- critical when missing churners "
        "is more costly than false alarms (which it is for B2B SaaS retention).",
    ])

    logger.info("\n%s", comparison.to_string(index=False))

    return comparison, "\n".join(md_lines)


# ===========================================================================
# 8. Business Interpretation Report (Markdown)
# ===========================================================================
def generate_business_report(
    comparison: pd.DataFrame,
    fi_rf: Optional[pd.DataFrame],
    fi_gb: Optional[pd.DataFrame],
    imbalance_report_md: str,
    y_test: pd.Series,
) -> str:
    """
    Generate a markdown business interpretation report.

    Covers:
        - What the best model's recall means in practice
        - What top features tell CS teams
        - Actionable recommendations
    """
    # Find best model by recall
    best_row = comparison.iloc[0]  # already sorted by recall desc
    best_name = best_row["model"]
    best_recall = best_row["recall_churn"]
    best_f1 = best_row["f1_churn"]
    best_auc = best_row["roc_auc"]
    best_precision = best_row["precision_churn"]

    n_churners = int(y_test.sum())
    n_total = len(y_test)
    detected = int(round(best_recall * n_churners))
    false_alarms = int(round(detected / best_precision)) - detected if best_precision > 0 else 0

    report = f"""# Stage 7 -- Model Evaluation Report
## B2B SaaS Churn Prediction

---

## 1. Model Comparison Summary

{comparison.to_markdown(index=False, floatfmt=".4f")}

> Sorted by **Recall (churn class) descending** -- catching churners is the
> business priority for ARR protection.

---

## 2. Best Model -- Business Interpretation

**Best model: `{best_name}`**
- Recall = **{best_recall:.1%}** -- We identify **{detected} out of {n_churners}
  churners** before they leave
- Precision = **{best_precision:.1%}** -- For every churner we correctly flag,
  we also flag ~{false_alarms} non-churners (acceptable cost for proactive outreach)
- ROC-AUC = **{best_auc:.3f}** -- The model ranks churners higher than non-churners
  {best_auc:.0%} of the time
- F1 = **{best_f1:.3f}** -- Balanced precision-recall tradeoff for the churn class

**What this means for the business:**
- In a portfolio of {n_total} accounts, we correctly flag **{detected}** at-risk
  accounts for proactive CS intervention
- The CS team should prioritize outreach to flagged accounts before contract renewal
- Even one saved enterprise account can protect $50K-$500K in ARR

---

## 3. Feature Importance -- What CS Teams Should Watch For

{interpret_feature_importances(fi_rf, fi_gb)}

**Actionable Signals for Customer Success:**
1. **Usage drops** (`usage_intensity`, `active_day_ratio`): Set up automated
   alerts when a customer's monthly usage drops >30% from baseline
2. **Inactivity** (`days_since_last_activity`): Flag accounts with >14 days
   of no product usage for immediate outreach
3. **Short tenure** (`tenure_months`): New customers (0-3 months) need
   structured onboarding and success milestones
4. **Support issues** (`ticket_count`, `avg_satisfaction`): Accounts with
   high ticket volume or low satisfaction need executive sponsor engagement
5. **Feature adoption** (`unique_features_used`): Low feature adoption
   means the customer hasn't found full product value -- offer training

---

## 4. Class Imbalance Analysis

{imbalance_report_md}

---

## 5. Recommendations

1. **Deploy `{best_name}`** as the production churn scoring model
2. **Set recall threshold >= 0.75** -- missing churners is more costly than
   false alarms in B2B SaaS
3. **Integrate predictions into CRM** -- surface churn risk scores alongside
   account health metrics
4. **Weekly scoring cadence** -- re-score all accounts weekly to catch
   emerging churn signals
5. **Feedback loop** -- track which flagged accounts actually churn to
   continuously improve the model

---

*Report generated by the Stage 7 evaluation pipeline.*
*Evaluation dataset: {n_total} accounts ({n_churners} churned, {n_total - n_churners} active).*
"""
    return report


# ===========================================================================
# Pipeline Entrypoint
# ===========================================================================
def run_evaluation(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "reports",
) -> Dict:
    """
    Full evaluation pipeline (Stage 7):
        1. Summary comparison DataFrame (sorted by Recall desc)
        2. Confusion matrices with TN/FP/FN/TP labels
        3. Classification reports
        4. ROC curves (overlaid)
        5. Precision-Recall curves (overlaid)
        6. Feature importances (top 15 for RF and GB)
        7. Class imbalance report (if X_train/y_train provided)
        8. Business interpretation markdown report

    Parameters
    ----------
    trained_models : dict
        {model_name: fitted model}
    X_test, y_test : test data
    X_train, y_train : training data (optional, used for imbalance report)
    feature_names : list of feature names
    output_dir : directory for reports

    Returns
    -------
    dict with comparison DataFrame, feature importances, and report path.
    """
    logger.info("=" * 60)
    logger.info("STARTING EVALUATION PIPELINE (STAGE 7)")
    logger.info("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Summary comparison ---
    comparison = evaluate_all_models(trained_models, X_test, y_test)
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (sorted by Recall — churn class)")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print()

    # --- 2 & 3. Per-model confusion matrices and classification reports ---
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        print_classification_report(y_test, y_pred, model_name=name, output_dir=output_dir)
        plot_confusion_matrix(y_test, y_pred, model_name=name)

    # --- 4. ROC curves ---
    plot_roc_curves(trained_models, X_test, y_test)

    # --- 5. Precision-Recall curves ---
    plot_precision_recall_curves(trained_models, X_test, y_test)

    # --- 6. Feature importances (top 15) ---
    if feature_names is None:
        feature_names = X_test.columns.tolist()

    fi_rf = None
    fi_gb = None
    for name in ["random_forest", "gradient_boosting"]:
        if name in trained_models:
            fi = plot_feature_importance(
                trained_models[name], feature_names,
                model_name=name, top_n=15,
            )
            if name == "random_forest":
                fi_rf = fi
            else:
                fi_gb = fi

    # Print feature importance interpretation
    interp = interpret_feature_importances(fi_rf, fi_gb)
    print(interp)

    # --- 7. Class imbalance report ---
    imbalance_report_md = ""
    imbalance_comparison = None
    if X_train is not None and y_train is not None:
        imbalance_comparison, imbalance_report_md = run_class_imbalance_report(
            X_train, X_test, y_train, y_test
        )
        print("\n" + imbalance_report_md)

    # --- 8. Business interpretation report ---
    report = generate_business_report(
        comparison, fi_rf, fi_gb, imbalance_report_md, y_test,
    )

    report_path = REPORTS_DIR / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved business evaluation report to %s", report_path)

    # Save comparison CSV
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info("Saved model comparison CSV to reports/model_comparison.csv")

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info("  Plots saved to: %s", FIGURES_DIR)
    logger.info("  Report saved to: %s", report_path)
    logger.info("=" * 60)

    return {
        "comparison": comparison,
        "fi_rf": fi_rf,
        "fi_gb": fi_gb,
        "imbalance_comparison": imbalance_comparison,
        "report_path": str(report_path),
    }


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse
    import joblib
    from src.feature_engineering import (
        build_preprocessor,
        BINARY_COLUMNS,
        ONEHOT_COLUMNS,
        TARGET_COL,
    )
    from src.train import split_data

    parser = argparse.ArgumentParser(description="Run evaluation pipeline (Stage 7)")
    parser.add_argument(
        "--models-dir", default="models",
        help="Directory with saved model .pkl files",
    )
    parser.add_argument(
        "--input", default="data/processed/abt_features.csv",
        help="Path to feature matrix CSV",
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="Directory for reports and plots",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Ensure types
    for col in BINARY_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(int)
    for col in ONEHOT_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Build preprocessor and split (same split as training for fair eval)
    preprocessor, _, _, _ = build_preprocessor(X)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Transform
    X_train_scaled = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index,
    )

    # Load trained models (versioned files)
    models_dir = Path(args.models_dir)
    trained_models = {}
    # Prefer versioned files, fallback to non-versioned
    for pkl_file in sorted(models_dir.glob("*_v1.pkl")):
        artifact = joblib.load(pkl_file)
        trained_models[artifact["model_name"]] = artifact["model"]

    if not trained_models:
        for pkl_file in sorted(models_dir.glob("*.pkl")):
            artifact = joblib.load(pkl_file)
            trained_models[artifact["model_name"]] = artifact["model"]

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Run evaluation
    results = run_evaluation(
        trained_models,
        X_test_scaled, y_test,
        X_train=X_train_scaled, y_train=y_train,
        feature_names=feature_names,
        output_dir=args.output_dir,
    )

    print(f"\n[OK] Evaluation complete! Report: {results['report_path']}")

"""
evaluate_models.py — Full Evaluation Module (Stage 7)
======================================================
Evaluates all four trained B2B SaaS churn prediction models:
    * DummyClassifier          (baseline floor)
    * LogisticRegression       (linear baseline, class_weight='balanced')
    * RandomForestClassifier   (tuned GridSearchCV)
    * GradientBoostingClassifier (tuned RandomizedSearchCV — production model)

Outputs (all plots -> reports/figures/):
    1.  Summary comparison DataFrame  -- sorted by Recall descending
    2.  Confusion matrix heatmaps     -- TN / FP / FN / TP labelled cells
    3.  ROC curves                    -- all models overlaid, AUC in legend
    4.  Precision-Recall curves       -- all models overlaid (imbalance-robust)
    5.  Feature importances           -- top-15 bar charts (RF + GB)
    6.  Class imbalance report        -- balanced vs unbalanced recall / F1
    7.  Markdown business report      -- saved to reports/evaluation_report.md

No SHAP -- sklearn feature_importances_ only.

Usage:
    python evaluate_models.py
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE_MODELS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "models" / "ml"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# Add project root to sys.path for local imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET_COL   = "churn_flag"

ONEHOT_COLUMNS = [
    "plan_tier", "country", "industry", "billing_frequency",
    "tenure_bucket", "mrr_segment", "referral_source",
]
BINARY_COLUMNS = [
    "is_trial", "upgrade_flag", "downgrade_flag",
    "auto_renew_flag", "near_renewal",
]

MODEL_DISPLAY_NAMES = {
    "dummy":               "Dummy (Baseline)",
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
}

FEATURE_INTERPRETATIONS = {
    "usage_intensity":          "Low events/month = disengaged -- strong churn signal",
    "tenure_months":            "Short tenure = hasn't seen full value yet -- high risk",
    "active_day_ratio":         "Sporadic usage = at-risk; stickiness not established",
    "days_since_last_activity": "Recent inactivity is a leading churn indicator",
    "avg_errors":               "High error rate degrades UX and erodes retention",
    "avg_duration_secs":        "Session length patterns reveal engagement depth",
    "mrr_amount":               "Revenue level -- different retention tactics by segment",
    "ticket_count":             "High support volume may signal product frustration",
    "avg_satisfaction":         "Low CSAT scores strongly correlate with churn intent",
    "min_satisfaction":         "Even one bad support interaction can trigger churn",
    "avg_resolution_hours":     "Slow resolution erodes customer confidence",
    "total_events":             "Overall engagement volume -- low = disengaged",
    "active_days":              "Fewer active days = low product stickiness",
    "seats":                    "Seat count -- enterprise accounts differ in churn dynamics",
    "avg_daily_usage":          "Low daily usage = product not embedded in workflow",
    "escalation_count":         "Escalations signal serious unresolved issues",
    "total_errors":             "Cumulative error rate high volume degrades experience",
    "unique_features_used":     "Low feature breadth = customer hasn't found full value",
    "beta_feature_usage":       "Beta adoption signals engaged, forward-leaning users",
    "avg_first_response_min":   "Slow first ticket response drives dissatisfaction",
    "near_renewal":             "Renewal milestone accounts are at a decision point",
}


# ===========================================================================
# 0. Data & model loading
# ===========================================================================

def _find_data_csv() -> Path:
    """Locate abt_features.csv; may live in data/interim or data/processed."""
    candidates = [
        PROJECT_ROOT / "data" / "interim"   / "abt_features.csv",
        PROJECT_ROOT / "data" / "processed" / "abt_features.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "abt_features.csv not found. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def load_data_and_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load abt_features.csv and produce the same 80/20 stratified split
    that was used during training (same random_state=42).
    """
    data_path = _find_data_csv()
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Enforce types expected by the ColumnTransformer
    for col in BINARY_COLUMNS:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)
    for col in ONEHOT_COLUMNS:
        if col in X.columns:
            X[col] = X[col].fillna("unknown").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Split -> Train: %d (churn=%.1f%%)  Test: %d (churn=%.1f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test),  y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessed_data(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Rebuild the same ColumnTransformer used in training by importing
    build_preprocessor from the project's feature_engineering module,
    fitting it on X_train only, then transforming both splits.

    Falls back to a numeric-only StandardScaler if the import fails.
    """
    _Xtr = X_train.copy()
    _Xte = X_test.copy()

    try:
        from src.features.feature_engineering import build_preprocessor  # type: ignore

        preprocessor, _, _, _ = build_preprocessor(_Xtr)
        Xtr_arr    = preprocessor.fit_transform(_Xtr)
        Xte_arr    = preprocessor.transform(_Xte)
        feat_names = list(preprocessor.get_feature_names_out())

        Xtr_sc = pd.DataFrame(Xtr_arr, columns=feat_names, index=_Xtr.index)
        Xte_sc = pd.DataFrame(Xte_arr, columns=feat_names, index=_Xte.index)
        logger.info("ColumnTransformer applied: %d features", len(feat_names))
        return Xtr_sc, Xte_sc, feat_names

    except Exception as e:
        logger.warning("build_preprocessor import failed (%s). Using numeric fallback.", e)
        num_cols = _Xtr.select_dtypes(include="number").columns.tolist()
        sc = StandardScaler()
        Xtr_sc = pd.DataFrame(
            sc.fit_transform(_Xtr[num_cols]), columns=num_cols, index=_Xtr.index
        )
        Xte_sc = pd.DataFrame(
            sc.transform(_Xte[num_cols]), columns=num_cols, index=_Xte.index
        )
        logger.info("Numeric-only fallback: %d features", len(num_cols))
        return Xtr_sc, Xte_sc, num_cols


def _train_models_fresh(
    X_train_sc: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """
    Train all four models from scratch with the same settings used in train.py.
    Used as fallback when stored pickle files are incompatible (numpy/sklearn version mismatch).
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    logger.info("Training all four models from scratch (pickle fallback) ...")

    models: Dict[str, Any] = {}

    # 1. Dummy
    logger.info("  Fitting DummyClassifier ...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_sc, y_train)
    models["dummy"] = dummy

    # 2. Logistic Regression
    logger.info("  Fitting LogisticRegression ...")
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE, solver="lbfgs",
    )
    lr.fit(X_train_sc, y_train)
    models["logistic_regression"] = lr

    # 3. Random Forest (Grid Search cv=5 scoring=f1)
    logger.info("  Fitting RandomForestClassifier (GridSearchCV) ...")
    rf_base = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_gs = GridSearchCV(
        estimator=rf_base,
        param_grid={"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        cv=5, scoring="f1", n_jobs=-1, refit=True, verbose=0,
    )
    rf_gs.fit(X_train_sc, y_train)
    models["random_forest"] = rf_gs.best_estimator_
    logger.info("    RF best params: %s", rf_gs.best_params_)

    # 4. Gradient Boosting (Randomized Search cv=5 scoring=roc_auc)
    logger.info("  Fitting GradientBoostingClassifier (RandomizedSearchCV) ...")
    gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_rs = RandomizedSearchCV(
        estimator=gb_base,
        param_distributions={"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 4, 5]},
        n_iter=10, cv=5, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0,
        random_state=RANDOM_STATE,
    )
    gb_rs.fit(X_train_sc, y_train)
    models["gradient_boosting"] = gb_rs.best_estimator_
    logger.info("    GB best params: %s", gb_rs.best_params_)

    logger.info("  All four models trained from scratch.")
    return models


def load_trained_models(
    X_train_sc: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Attempt to load the four versioned model pkl files.
    If loading fails (numpy/sklearn version mismatch), retrain from scratch.
    X_train_sc and y_train are required for the fallback case.
    """
    model_files = {
        "dummy":               MODELS_DIR / "dummy_v1.pkl",
        "logistic_regression": MODELS_DIR / "logistic_regression_v1.pkl",
        "random_forest":       MODELS_DIR / "random_forest_v1.pkl",
        "gradient_boosting":   MODELS_DIR / "gradient_boosting_v1.pkl",
    }
    trained: Dict[str, Any] = {}
    failed = []

    for name, path in model_files.items():
        if not path.exists():
            logger.warning("Model pkl not found: %s", path)
            failed.append(name)
            continue
        try:
            artifact = joblib.load(path)
            trained[name] = artifact["model"] if isinstance(artifact, dict) else artifact
            logger.info("Loaded  %-25s  (%s)", name, type(trained[name]).__name__)
        except Exception as e:
            logger.warning("Cannot load %s (%s) -- will retrain.", name, e)
            failed.append(name)

    if failed:
        if X_train_sc is None or y_train is None:
            raise RuntimeError(
                f"Pkl load failed for {failed} and no training data provided for fallback."
            )
        logger.info("Retraining all models from scratch (version mismatch fallback) ...")
        trained = _train_models_fresh(X_train_sc, y_train)

    return trained



# ===========================================================================
# 1. Metrics
# ===========================================================================

def compute_metrics(
    y_true:     pd.Series,
    y_pred:     np.ndarray,
    y_proba:    Optional[np.ndarray],
    model_key:  str,
) -> Dict[str, Any]:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    roc_val = brier = float("nan")
    if y_proba is not None:
        try:
            roc_val = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass
        try:
            brier = brier_score_loss(y_true, y_proba)
        except ValueError:
            pass

    return {
        "Model":             MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "_key":              model_key,
        "Accuracy":          round(acc,    4),
        "Precision (churn)": round(prec,   4),
        "Recall (churn)":    round(rec,    4),
        "F1 (churn)":        round(f1,     4),
        "ROC-AUC":           round(roc_val, 4) if not np.isnan(roc_val) else float("nan"),
        "Brier Score":       round(brier,   4) if not np.isnan(brier)   else float("nan"),
        "_y_pred":           y_pred,
        "_y_proba":          y_proba,
    }


def build_comparison_df(
    trained_models: Dict[str, Any],
    X_test_sc:      pd.DataFrame,
    y_test:         pd.Series,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Predict with every model on the pre-processed test set.
    Returns (comparison_df sorted by Recall desc, raw rows list).
    """
    rows = []
    for key, model in trained_models.items():
        try:
            y_pred = model.predict(X_test_sc)
        except Exception as ex:
            logger.warning("predict failed for %s (%s). Trying numeric subset.", key, ex)
            num_cols = X_test_sc.select_dtypes(include="number").columns
            y_pred   = model.predict(X_test_sc[num_cols])

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_sc)[:, 1]
            except Exception as ex:
                logger.warning("predict_proba failed for %s (%s).", key, ex)
                try:
                    num_cols = X_test_sc.select_dtypes(include="number").columns
                    y_proba  = model.predict_proba(X_test_sc[num_cols])[:, 1]
                except Exception:
                    pass

        rows.append(compute_metrics(y_test, y_pred, y_proba, model_key=key))

    # Build display DataFrame
    dcols = ["Model", "Accuracy", "Precision (churn)", "Recall (churn)", "F1 (churn)", "ROC-AUC"]
    df = (
        pd.DataFrame(rows)[dcols + ["Brier Score", "_key", "_y_pred", "_y_proba"]]
        .sort_values("Recall (churn)", ascending=False)
        .reset_index(drop=True)
    )
    return df, rows


# ===========================================================================
# 2. Confusion matrices
# ===========================================================================

def plot_confusion_matrices(rows: List[Dict], y_test: pd.Series) -> None:
    """Seaborn heatmap per model with TN / FP / FN / TP labelled cells."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for row, color in zip(rows, PALETTE_MODELS):
        name    = row["_key"]
        y_pred  = row["_y_pred"]
        display = row["Model"]

        cm              = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp  = cm.ravel()
        labels          = np.array([
            [f"True Negative\n{tn:,}",  f"False Positive\n{fp:,}"],
            [f"False Negative\n{fn:,}", f"True Positive\n{tp:,}"],
        ])

        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues",
            xticklabels=["Not Churned", "Churned"],
            yticklabels=["Not Churned", "Churned"],
            ax=ax, linewidths=0.6,
            cbar_kws={"label": "Count", "shrink": 0.85},
            annot_kws={"size": 13, "fontweight": "bold"},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix -- {display}", fontsize=14, fontweight="bold", pad=15)
        fig.tight_layout()
        path = FIGURES_DIR / f"cm_{name}.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved confusion matrix -> %s", path)


# ===========================================================================
# 3. ROC curves
# ===========================================================================

def plot_roc_curves(rows: List[Dict], y_test: pd.Series) -> None:
    """All-models ROC curve overlay with AUC in legend."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    for row, color in zip(rows, PALETTE_MODELS):
        label   = row["Model"]
        y_proba = row["_y_proba"]
        roc_val = row["ROC-AUC"]

        if y_proba is not None and not np.isnan(roc_val):
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, color=color, lw=2.2,
                    label=f"{label}  (AUC = {auc(fpr, tpr):.3f})")
        else:
            ax.plot([], [], color=color, lw=2.2, label=f"{label}  (no probability)")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random Chance  (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves -- All Models\n(Higher & further left = better)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "roc_curves_all_models.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved ROC curves -> %s", path)


# ===========================================================================
# 4. Precision-Recall curves
# ===========================================================================

def plot_precision_recall_curves(rows: List[Dict], y_test: pd.Series) -> None:
    """
    Overlaid PR curves -- more informative than ROC under class imbalance.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = float(y_test.mean())

    for row, color in zip(rows, PALETTE_MODELS):
        label   = row["Model"]
        y_proba = row["_y_proba"]

        if y_proba is not None:
            precs, recs, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recs, precs)
            ax.plot(recs, precs, color=color, lw=2.2,
                    label=f"{label}  (PR-AUC = {pr_auc:.3f})")
        else:
            ax.scatter(
                [row["Recall (churn)"]], [row["Precision (churn)"]],
                color=color, s=140, zorder=5,
                label=f"{label}  (threshold point)",
            )

    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1.4,
               label=f"No-skill  (prevalence = {baseline:.3f})")
    ax.fill_between([0, 1], [baseline, baseline], [1, 1], alpha=0.03, color="green")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves -- All Models\n"
        "(More informative than ROC under class imbalance)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "precision_recall_curves.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved PR curves -> %s", path)


# ===========================================================================
# 5. Feature importances
# ===========================================================================

def plot_feature_importance(
    model:         Any,
    feature_names: List[str],
    model_key:     str,
    display_name:  str,
    top_n:         int = 15,
) -> Optional[pd.DataFrame]:
    """
    Horizontal bar chart of top-N sklearn feature_importances_.
    Strips ColumnTransformer prefix (num__, cat__, bin__) from labels.
    Returns DataFrame of importances sorted desc.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("'%s' has no feature_importances_ -- skipping.", model_key)
        return None

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        logger.warning(
            "Feature mismatch for %s: %d importances vs %d names -- trimming.",
            model_key, len(importances), len(feature_names)
        )
        min_len      = min(len(importances), len(feature_names))
        importances  = importances[:min_len]
        feature_names = feature_names[:min_len]

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Clean display labels (strip transformer prefix)
    fi_df["label"] = fi_df["feature"].apply(
        lambda f: f.split("__")[-1] if "__" in f else f
    )

    color_map = {"random_forest": "#55A868", "gradient_boosting": "#C44E52"}
    bar_color = color_map.get(model_key, "#4C72B0")

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.47)))
    bars = ax.barh(
        fi_df["label"][::-1],
        fi_df["importance"][::-1],
        color=bar_color, alpha=0.88, edgecolor="white",
    )
    for bar, val in zip(bars, fi_df["importance"][::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color="#333333")

    ax.set_xlabel("Feature Importance (Gini / Mean Decrease Impurity)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} Features -- {display_name}", fontsize=14,
                 fontweight="bold", pad=14)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"feature_importance_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved feature importance -> %s", path)
    return fi_df


def interpret_feature_importances(
    fi_rf: Optional[pd.DataFrame],
    fi_gb: Optional[pd.DataFrame],
    top_n: int = 5,
) -> str:
    """Generate markdown interpretation of top SaaS churn features."""
    lines = ["### Feature Importance Interpretation\n"]

    def _clean(feat: str) -> str:
        return feat.split("__")[-1] if "__" in feat else feat

    for label, fi_df in [("Random Forest", fi_rf), ("Gradient Boosting", fi_gb)]:
        if fi_df is None:
            continue
        lines.append(f"**{label} -- Top {top_n} features:**\n")
        for _, row in fi_df.head(top_n).iterrows():
            clean = _clean(row["feature"])
            interp = FEATURE_INTERPRETATIONS.get(clean, "Predictive signal -- investigate further")
            lines.append(f"  - `{row['feature']}` ({row['importance']:.4f}):  {interp}")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# 6. Class imbalance report
# ===========================================================================

def run_class_imbalance_report(
    X_train_raw: pd.DataFrame,
    X_test_raw:  pd.DataFrame,
    y_train:     pd.Series,
    y_test:      pd.Series,
) -> Tuple[pd.DataFrame, str]:
    """
    Train quick LogReg / RF / GB models with and without class_weight='balanced'
    on numeric features only.  Compare Recall (churn) and F1 (churn).
    Returns (comparison_df, markdown_string).
    """
    logger.info("Running class imbalance comparison ...")
    num_cols = X_train_raw.select_dtypes(include="number").columns.tolist()
    Xtr = X_train_raw[num_cols]
    Xte = X_test_raw[num_cols]

    configs = {
        "LogReg  (unbalanced)":        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"),
        "LogReg  (balanced)":          LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs"),
        "RandomForest  (unbalanced)":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "RandomForest  (balanced)":    RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "GradBoost  (unbalanced)":     GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GradBoost  (sample_weight)":  GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    rows = []
    for cfg_name, model in configs.items():
        sw = None
        if "sample_weight" in cfg_name:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            wp = len(y_train) / (2.0 * n_pos) if n_pos > 0 else 1.0
            wn = len(y_train) / (2.0 * n_neg) if n_neg > 0 else 1.0
            sw = y_train.map({0: wn, 1: wp}).values

        fit_kwargs = {"sample_weight": sw} if sw is not None else {}
        model.fit(Xtr, y_train, **fit_kwargs)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

        rows.append({
            "Configuration":     cfg_name,
            "Recall (churn)":    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            "Precision (churn)": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            "F1 (churn)":        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            "Accuracy":          round(accuracy_score(y_test, y_pred), 4),
            "ROC-AUC":           round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else float("nan"),
        })

    cdf = (
        pd.DataFrame(rows)
        .sort_values("Recall (churn)", ascending=False)
        .reset_index(drop=True)
    )

    md = [
        "### Class Imbalance Handling Report\n",
        "Comparison **with** and **without** `class_weight='balanced'` "
        "(or `sample_weight` for Gradient Boosting):\n",
        cdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        "**Key Findings:**",
    ]
    for base, bkw, ukw in [
        ("LogReg",       r"LogReg.*balanced",      r"LogReg.*unbalanced"),
        ("RandomForest", r"RandomForest.*balanced", r"RandomForest.*unbalanced"),
        ("GradBoost",    r"GradBoost.*sample",      r"GradBoost.*unbalanced"),
    ]:
        brow = cdf[cdf["Configuration"].str.contains(bkw, case=False, regex=True)]
        urow = cdf[cdf["Configuration"].str.contains(ukw, case=False, regex=True)]
        if not brow.empty and not urow.empty:
            dr = float(brow["Recall (churn)"].iloc[0]) - float(urow["Recall (churn)"].iloc[0])
            df_ = float(brow["F1 (churn)"].iloc[0])    - float(urow["F1 (churn)"].iloc[0])
            md.append(f"- **{base}**: balancing -> Recall delta={dr:+.3f} | F1 delta={df_:+.3f}")

    md.extend([
        "",
        "> **Insight**: Class balancing trades raw accuracy for recall -- exactly the right "
        "trade-off for B2B SaaS churn. Missing a churner costs 12x more in ARR than a "
        "false-alarm outreach call.",
    ])

    _plot_imbalance_bars(cdf)
    return cdf, "\n".join(md)


def _plot_imbalance_bars(cdf: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, metric in zip(axes, ["Recall (churn)", "F1 (churn)"]):
        colors = [
            "#C44E52" if ("balanced" in c or "sample" in c) else "#4C72B0"
            for c in cdf["Configuration"]
        ]
        bars = ax.barh(cdf["Configuration"], cdf[metric], color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, cdf[metric]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"{metric} by Configuration", fontsize=13, fontweight="bold")
        ax.set_xlim([0, 1.12])
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(x=0.75, color="green", linestyle="--", alpha=0.7, lw=1.2, label="Target >= 0.75")
        ax.legend(fontsize=8)

    bal_p   = mpatches.Patch(color="#C44E52", alpha=0.85, label="With balancing")
    unbal_p = mpatches.Patch(color="#4C72B0", alpha=0.85, label="Without balancing")
    fig.legend(handles=[bal_p, unbal_p], loc="lower center", ncol=2,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Effect of Class Balancing on Recall and F1 (Churn Class)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = FIGURES_DIR / "class_imbalance_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved imbalance comparison -> %s", path)


# ===========================================================================
# 7. Model comparison bar chart
# ===========================================================================

def plot_model_comparison_summary(comparison_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics  = ["Accuracy", "Precision (churn)", "Recall (churn)", "F1 (churn)", "ROC-AUC"]
    df_plot  = comparison_df[["Model"] + metrics].copy()
    model_list = df_plot["Model"].tolist()

    fig, ax = plt.subplots(figsize=(13, 6))
    x, width = np.arange(len(metrics)), 0.18

    for i, (mname, color) in enumerate(zip(model_list, PALETTE_MODELS)):
        vals = df_plot[df_plot["Model"] == mname][metrics].values.flatten()
        bars = ax.bar(x + i * width, vals, width, label=mname,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x + width * (len(model_list) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0, 1.18])
    ax.set_title("Model Comparison -- All Metrics", fontsize=14, fontweight="bold")
    ax.axhline(y=0.75, color="grey", linestyle="--", alpha=0.5, lw=1, label="0.75 threshold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "model_comparison_summary.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved comparison summary -> %s", path)


# ===========================================================================
# 8. Business interpretation report
# ===========================================================================

def generate_business_report(
    comparison_df:  pd.DataFrame,
    fi_rf:          Optional[pd.DataFrame],
    fi_gb:          Optional[pd.DataFrame],
    imbalance_md:   str,
    y_test:         pd.Series,
) -> str:
    best         = comparison_df.iloc[0]
    best_name    = best["Model"]
    best_recall  = float(best["Recall (churn)"])
    best_prec    = float(best["Precision (churn)"])
    best_f1      = float(best["F1 (churn)"])
    best_auc     = float(best["ROC-AUC"])

    n_curn   = int(y_test.sum())
    n_total  = len(y_test)
    detected = int(round(best_recall * n_curn))
    fa       = (int(round(detected / best_prec)) - detected) if best_prec > 0 else "N/A"

    feat_md = interpret_feature_importances(fi_rf, fi_gb, top_n=5)

    dcols = ["Model", "Accuracy", "Precision (churn)", "Recall (churn)", "F1 (churn)", "ROC-AUC"]

    return f"""# Stage 7 -- Model Evaluation Report
## B2B SaaS Churn Prediction (RavenStack CRM Dataset)

*Generated by `evaluate_models.py` | Test set: {n_total:,} accounts ({n_curn:,} churned, {n_total - n_curn:,} active)*

---

## 1. Model Comparison Summary

> **Sort order: Recall (churn) descending** -- catching churners before they leave is the
> primary business objective. Missing a churner costs 12-50x more than a false-alarm call.

{comparison_df[dcols].to_markdown(index=False, floatfmt=".4f")}

---

## 2. Best Model -- Business Interpretation

**Best model (by Recall): `{best_name}`**

| Metric | Score | Business Meaning |
|--------|-------|-----------------|
| **Recall** | **{best_recall:.1%}** | We identify **{detected} of {n_curn} churners** before they leave |
| **Precision** | **{best_prec:.1%}** | For every churner flagged, ~{fa} non-churners also flagged |
| **ROC-AUC** | **{best_auc:.3f}** | Ranks churners above non-churners {best_auc:.0%} of the time |
| **F1** | **{best_f1:.3f}** | Balanced precision-recall for the churn class |

### What this means for the CS team:

- In a portfolio of **{n_total:,} accounts**, we correctly flag **{detected:,} at-risk accounts**
  for proactive outreach *before* they churn.
- The ~{fa} false alarms are an acceptable cost -- a 15-minute check-in call costs
  far less than losing a renewal contract worth $50K-$500K in ARR.
- We leave only **{n_curn - detected} churners undetected** -- "dark churn" reducible
  through lower probability thresholds and richer signals.

---

## 3. Feature Importances -- What CS Teams Should Watch For

{feat_md}

### Actionable CS Playbook:

| # | Signal | Trigger | CS Action |
|---|--------|---------|-----------|
| 1 | Usage drop (`usage_intensity`, `active_day_ratio`) | >30% MoM decline | Automated alert + 48h outreach |
| 2 | Inactivity (`days_since_last_activity`) | >14 days no login | Immediate value walkthrough offer |
| 3 | Short tenure (`tenure_months`) | 0-3 months | Structured onboarding milestones |
| 4 | Support friction (`ticket_count`, `avg_satisfaction`) | High volume or CSAT <3 | Executive sponsor escalation |
| 5 | Low adoption (`unique_features_used`) | <3 core features used | Feature discovery session |
| 6 | Near renewal (`near_renewal`) | <30 days to renewal | Proactive QBR + renewal incentive |

---

## 4. Class Imbalance Analysis

{imbalance_md}

---

## 5. Recommendations

1. **Deploy `{best_name}`** as the production churn scoring model.
2. **Lower probability threshold** (e.g., 0.35-0.40) to push recall above 0.80 at cost of precision.
3. **Weekly batch scoring** -- re-score all active accounts; surface risk in CRM.
4. **Feedback loop** -- track which flagged accounts actually churn; retrain quarterly.
5. **Segment models** -- build separate models for enterprise vs SMB; churn drivers differ.

---

## 6. Evaluation Artefacts (`reports/figures/`)

| File | Description |
|------|-------------|
| `cm_*.png` | Confusion matrices -- TN/FP/FN/TP per model |
| `roc_curves_all_models.png` | ROC overlay -- all 4 models |
| `precision_recall_curves.png` | PR curve overlay -- all 4 models |
| `feature_importance_random_forest.png` | Top-15 RF importances |
| `feature_importance_gradient_boosting.png` | Top-15 GB importances |
| `class_imbalance_comparison.png` | Balanced vs unbalanced recall / F1 |
| `model_comparison_summary.png` | All metrics grouped bar chart |

---
*All models are pure scikit-learn -- no XGBoost, LightGBM, or SHAP.*
"""


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_full_evaluation() -> Dict[str, Any]:
    """
    Orchestrate all seven evaluation stages and return a results dict.
    """
    logger.info("=" * 65)
    logger.info("  STAGE 7: FULL MODEL EVALUATION PIPELINE")
    logger.info("=" * 65)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 0. Load data -------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = load_data_and_split()

    # ---- Pre-process data (ColumnTransformer / StandardScaler on train) -----
    X_train_sc, X_test_sc, feat_names = build_preprocessed_data(X_train_raw, X_test_raw)

    # ---- Load trained models (fallback: retrain with X_train_sc) -----------
    trained_models = load_trained_models(X_train_sc=X_train_sc, y_train=y_train)

    if not trained_models:
        raise RuntimeError(
            f"No models available. Check {MODELS_DIR} or allow retraining."
        )
    logger.info("Models ready: %s", list(trained_models.keys()))


    # ---- 1. Summary comparison table ------------------------------------
    logger.info("[1/7] Building comparison table ...")
    comparison_df, rows = build_comparison_df(trained_models, X_test_sc, y_test)

    dcols = ["Model", "Accuracy", "Precision (churn)", "Recall (churn)", "F1 (churn)", "ROC-AUC"]
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON  (sorted by Recall -- churn class, descending)")
    print("=" * 72)
    print(comparison_df[dcols].to_string(index=False))
    print()

    # Print classification reports
    print("\n" + "-" * 72)
    print("  CLASSIFICATION REPORTS (per model)")
    print("-" * 72)
    for row in rows:
        y_pred = row["_y_pred"]
        print(f"\n--- {row['Model']} ---")
        print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))

    # ---- 2. Confusion matrices ------------------------------------------
    logger.info("[2/7] Plotting confusion matrices ...")
    plot_confusion_matrices(rows, y_test)

    # ---- 3. ROC curves -------------------------------------------------
    logger.info("[3/7] Plotting ROC curves ...")
    plot_roc_curves(rows, y_test)

    # ---- 4. PR curves --------------------------------------------------
    logger.info("[4/7] Plotting Precision-Recall curves ...")
    plot_precision_recall_curves(rows, y_test)

    # ---- 5. Feature importances ----------------------------------------
    logger.info("[5/7] Plotting feature importances (RF + GB top 15) ...")
    fi_rf = fi_gb = None
    for key in ["random_forest", "gradient_boosting"]:
        if key not in trained_models:
            continue
        model   = trained_models[key]
        display = MODEL_DISPLAY_NAMES.get(key, key)
        # feat_names comes from the freshly-built preprocessor; use directly
        fi_feat_names = feat_names

        fi = plot_feature_importance(model, fi_feat_names, key, display, top_n=15)
        if key == "random_forest":
            fi_rf = fi
        else:
            fi_gb = fi

    print("\n" + interpret_feature_importances(fi_rf, fi_gb, top_n=5))

    # ---- 6. Class imbalance report ------------------------------------
    logger.info("[6/7] Class imbalance handling report ...")
    imbalance_df, imbalance_md = run_class_imbalance_report(
        X_train_raw, X_test_raw, y_train, y_test
    )
    print("\n" + imbalance_md)

    # ---- Bonus: model comparison bar chart ----------------------------
    plot_model_comparison_summary(comparison_df)

    # ---- 7. Business report -------------------------------------------
    logger.info("[7/7] Writing business interpretation report ...")
    report_md   = generate_business_report(comparison_df, fi_rf, fi_gb, imbalance_md, y_test)
    report_path = REPORTS_DIR / "evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved report -> %s", report_path)

    csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df[dcols].to_csv(csv_path, index=False)
    logger.info("Saved CSV   -> %s", csv_path)

    logger.info("=" * 65)
    logger.info("  EVALUATION COMPLETE")
    logger.info("  Figures  -> %s", FIGURES_DIR)
    logger.info("  Report   -> %s", report_path)
    logger.info("  CSV      -> %s", csv_path)
    logger.info("=" * 65)

    print(f"\n[OK] Evaluation complete!")
    print(f"   Figures -> {FIGURES_DIR}")
    print(f"   Report  -> {report_path}")
    print(f"   CSV     -> {csv_path}")

    return {
        "comparison":    comparison_df,
        "rows":          rows,
        "fi_rf":         fi_rf,
        "fi_gb":         fi_gb,
        "imbalance_df":  imbalance_df,
        "report_path":   str(report_path),
    }


if __name__ == "__main__":
    run_full_evaluation()

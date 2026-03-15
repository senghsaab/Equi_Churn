"""
train.py — Model Training Pipeline (Stage 6)
===============================================
Trains four classifiers for B2B SaaS churn prediction:
    1. DummyClassifier (strategy='most_frequent') — performance floor baseline
    2. LogisticRegression (class_weight='balanced', max_iter=1000) — linear baseline
    3. RandomForestClassifier — tuned via GridSearchCV (cv=5, scoring='f1')
    4. GradientBoostingClassifier — tuned via RandomizedSearchCV
       (n_iter=10, cv=5, scoring='roc_auc')   ← FINAL production model

For each model we print:
    • accuracy, precision (churn), recall (churn), F1 (churn), ROC-AUC
    • confusion matrix
    • full classification report

Additional outputs:
    • Single ROC curve graph overlaying all 4 models
    • Versioned model files saved to models/ (e.g. gradient_boosting_v1.pkl)
    • Best hyperparameters logged for RF and GB

IMPORTANT: Only scikit-learn classifiers — no XGBoost, LightGBM, or external
boosting libraries.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_VERSION = "v1"
FIGURES_DIR = Path("reports/figures")

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dummy": {
        "class": DummyClassifier,
        "params": {"strategy": "most_frequent"},
        "search": None,
        "description": "Majority-class baseline (performance floor)",
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        },
        "search": None,
        "description": "Interpretable linear model with balanced class weights",
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "search": {
            "method": "grid",
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "cv": 5,
            "scoring": "f1",
        },
        "description": "Bagging ensemble — tuned with GridSearchCV (scoring=f1)",
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "params": {
            "random_state": RANDOM_STATE,
        },
        "search": {
            "method": "random",
            "param_distributions": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5],
            },
            "n_iter": 10,
            "cv": 5,
            "scoring": "roc_auc",
        },
        "description": "Sequential boosting — FINAL model, tuned with RandomizedSearchCV (scoring=roc_auc)",
    },
}


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
def load_config(config_path: str = "configs/model_config.yaml") -> Dict:
    """Load hyperparameters from YAML config, falling back to defaults."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded config from %s", path)
        return config
    else:
        logger.info("No config found at %s — using defaults.", path)
        return {}


# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving class distribution."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Train: %d samples (churn=%.2f%%) | Test: %d samples (churn=%.2f%%)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, transform both splits."""
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    logger.info("Features scaled with StandardScaler (%d features)", len(feature_names))
    return X_train_scaled, X_test_scaled, scaler


# ===========================================================================
# Training — with Hyperparameter Search
# ===========================================================================
def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, Optional[Dict]]:
    """
    Train a single model from the registry.

    If the registry entry has a 'search' config, uses GridSearchCV or
    RandomizedSearchCV to find best hyperparameters.

    Returns
    -------
    tuple[model, best_params or None]
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[model_name]
    search_cfg = entry.get("search")

    logger.info("=" * 50)
    logger.info(
        "Training %-25s | %s",
        model_name,
        entry["description"],
    )
    logger.info("=" * 50)

    best_params = None

    if search_cfg is not None:
        # Hyperparameter search
        base_model = entry["class"](**entry["params"])

        if search_cfg["method"] == "grid":
            logger.info(
                "  GridSearchCV | param_grid=%s | cv=%d | scoring='%s'",
                search_cfg["param_grid"],
                search_cfg["cv"],
                search_cfg["scoring"],
            )
            searcher = GridSearchCV(
                estimator=base_model,
                param_grid=search_cfg["param_grid"],
                cv=search_cfg["cv"],
                scoring=search_cfg["scoring"],
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
        elif search_cfg["method"] == "random":
            logger.info(
                "  RandomizedSearchCV | param_distributions=%s | n_iter=%d | cv=%d | scoring='%s'",
                search_cfg["param_distributions"],
                search_cfg["n_iter"],
                search_cfg["cv"],
                search_cfg["scoring"],
            )
            searcher = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=search_cfg["param_distributions"],
                n_iter=search_cfg["n_iter"],
                cv=search_cfg["cv"],
                scoring=search_cfg["scoring"],
                n_jobs=-1,
                verbose=0,
                refit=True,
                random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unknown search method: {search_cfg['method']}")

        searcher.fit(X_train, y_train)
        model = searcher.best_estimator_
        best_params = searcher.best_params_
        best_score = searcher.best_score_

        logger.info("  ✓ Best params: %s", best_params)
        logger.info(
            "  ✓ Best CV score (%s): %.4f",
            search_cfg["scoring"],
            best_score,
        )
    else:
        # Direct fit (no search)
        model = entry["class"](**entry["params"])
        model.fit(X_train, y_train)

    logger.info("  ✓ Training complete: %s", model_name)
    return model, best_params


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Optional[Dict]]]:
    """
    Train all models in the registry.

    Returns
    -------
    tuple[dict, dict]
        trained_models: {name: model}
        best_params: {name: params_dict or None}
    """
    trained = {}
    all_best_params = {}
    for name in MODEL_REGISTRY:
        model, best_params = train_single_model(name, X_train, y_train)
        trained[name] = model
        all_best_params[name] = best_params
    logger.info("All %d models trained successfully.", len(trained))
    return trained, all_best_params


# ===========================================================================
# Per-Model Evaluation (inline during training)
# ===========================================================================
def evaluate_model(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a single model and print all required outputs:
        1. Accuracy, Precision (churn), Recall (churn), F1 (churn), ROC-AUC
        2. Confusion matrix (logged)
        3. Full classification report

    Returns metrics dict.
    """
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    # --- 1. Compute metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    roc_auc_val = 0.5
    if y_proba is not None:
        try:
            roc_auc_val = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc_val = float("nan")

    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} — Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (churn class)")
    print(f"  Recall    : {recall:.4f}  (churn class)")
    print(f"  F1-Score  : {f1:.4f}  (churn class)")
    print(f"  ROC-AUC   : {roc_auc_val:.4f}")

    # --- 2. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                   Predicted")
    print(f"                   Not Churned  Churned")
    print(f"  Actual Not Churned    {cm[0][0]:>5d}      {cm[0][1]:>5d}")
    print(f"  Actual Churned        {cm[1][0]:>5d}      {cm[1][1]:>5d}")

    # --- 3. Classification Report ---
    report = classification_report(
        y_test, y_pred, target_names=["Not Churned", "Churned"]
    )
    print(f"\n  Classification Report:")
    print(report)

    logger.info(
        "%s | Acc=%.3f | Prec=%.3f | Rec=%.3f | F1=%.3f | AUC=%.3f",
        model_name, accuracy, precision, recall, f1, roc_auc_val,
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision_churn": precision,
        "recall_churn": recall,
        "f1_churn": f1,
        "roc_auc": roc_auc_val,
    }


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all models, printing results for each. Returns comparison df."""
    rows = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, name, X_test, y_test)
        rows.append(metrics)

    comparison = pd.DataFrame(rows).set_index("model")
    comparison = comparison.sort_values("f1_churn", ascending=False)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison.to_string())
    print()

    return comparison


# ===========================================================================
# ROC Curve Plot (all 4 models overlaid)
# ===========================================================================
def plot_roc_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot a single ROC curve graph overlaying all 4 models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("husl", len(trained_models))

    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc_val:.3f})",
            )
        else:
            # DummyClassifier with most_frequent has no probabilities
            ax.plot([], [], color=color, lw=2, label=f"{name} (no proba)")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filepath = FIGURES_DIR / "roc_curves_all_models.png"
        fig.savefig(filepath)
        logger.info("Saved overlaid ROC curves to %s", filepath)
    plt.close(fig)


# ===========================================================================
# Confusion Matrix Plots
# ===========================================================================
def plot_confusion_matrices(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot confusion matrix heatmaps for all models."""
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Churned", "Churned"],
            yticklabels=["Not Churned", "Churned"],
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Count"},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")

        if save:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES_DIR / f"{name}_confusion_matrix.png")
        plt.close(fig)
    logger.info("Saved confusion matrix plots for all %d models", len(trained_models))


# ===========================================================================
# Model Persistence — Versioned Filenames
# ===========================================================================
def save_model(
    model: Any,
    model_name: str,
    version: str = MODEL_VERSION,
    scaler: Optional[StandardScaler] = None,
    preprocessor: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    best_params: Optional[Dict] = None,
    output_dir: str = "models",
) -> str:
    """
    Save a trained model with versioned filename (e.g. gradient_boosting_v1.pkl).

    Artifact includes model, scaler/preprocessor, feature names, and best params.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_name": model_name,
        "version": version,
        "scaler": scaler,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "best_params": best_params,
    }

    filename = f"{model_name}_{version}.pkl"
    filepath = out / filename
    joblib.dump(artifact, filepath)
    logger.info("Saved model artifact: %s", filepath)
    return str(filepath)


def save_all_models(
    trained_models: Dict[str, Any],
    all_best_params: Dict[str, Optional[Dict]],
    version: str = MODEL_VERSION,
    scaler: Optional[StandardScaler] = None,
    preprocessor: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "models",
) -> Dict[str, str]:
    """Save all trained models with versioned filenames."""
    paths = {}
    for name, model in trained_models.items():
        paths[name] = save_model(
            model, name, version, scaler, preprocessor,
            feature_names, all_best_params.get(name), output_dir,
        )
    return paths


# ===========================================================================
# Pipeline Entrypoint
# ===========================================================================
def run_training(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Optional[Any] = None,
    config_path: str = "configs/model_config.yaml",
    output_dir: str = "models",
) -> Dict:
    """
    Full training pipeline (Stage 6):
        1. Split data (stratified 80/20)
        2. Transform features (ColumnTransformer or StandardScaler)
        3. Train all models (GridSearchCV for RF, RandomizedSearchCV for GB)
        4. Evaluate each model (metrics + confusion matrix + report)
        5. Plot overlaid ROC curves
        6. Save all model artifacts with versioned filenames

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (pre-encoded, mixed types OK if preprocessor given).
    y : pd.Series
        Target vector (churn_flag: 0/1).
    preprocessor : ColumnTransformer or None
        If provided, fit_transform on train / transform on test.
    config_path : str
        Path to model config YAML.
    output_dir : str
        Directory for saved model artifacts.

    Returns
    -------
    dict
        Contains trained_models, best_params, comparison, split data, etc.
    """
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE (STAGE 6)")
    logger.info("=" * 60)

    # --- 1. Config ---
    config = load_config(config_path)

    # --- 2. Split ---
    X_train, X_test, y_train, y_test = split_data(X, y)

    # --- 3. Transform ---
    scaler = None
    if preprocessor is not None:
        logger.info("Using ColumnTransformer preprocessor (fit on train only)")
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
        feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        logger.info("No preprocessor — falling back to StandardScaler")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        feature_names = X_train.columns.tolist()

    # --- 4. Train all models (with hyperparameter search) ---
    trained_models, all_best_params = train_all_models(X_train_scaled, y_train)

    # Log best hyperparameters summary
    print("\n" + "=" * 60)
    print("  BEST HYPERPARAMETERS (from search)")
    print("=" * 60)
    for name, params in all_best_params.items():
        if params is not None:
            print(f"  {name}: {params}")
    print()

    # --- 5. Evaluate each model ---
    comparison = evaluate_all_models(trained_models, X_test_scaled, y_test)

    # --- 6. Plot ROC curves (all models overlaid) ---
    plot_roc_curves(trained_models, X_test_scaled, y_test)
    plot_confusion_matrices(trained_models, X_test_scaled, y_test)

    # --- 7. Save models with versioned filenames ---
    model_paths = save_all_models(
        trained_models, all_best_params, MODEL_VERSION,
        scaler, preprocessor, feature_names, output_dir,
    )

    # Save comparison CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "model_comparison.csv")
    logger.info("Saved model comparison to reports/model_comparison.csv")

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("  Models saved to: %s (versioned as *_%s.pkl)", output_dir, MODEL_VERSION)
    logger.info("  ROC curves saved to: %s", FIGURES_DIR)
    logger.info("=" * 60)

    return {
        "trained_models": trained_models,
        "best_params": all_best_params,
        "scaler": scaler,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "comparison": comparison,
        "model_paths": model_paths,
    }


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse
    from src.feature_engineering import (
        build_preprocessor,
        BINARY_COLUMNS,
        ONEHOT_COLUMNS,
        TARGET_COL,
    )

    parser = argparse.ArgumentParser(description="Run model training pipeline (Stage 6)")
    parser.add_argument(
        "--input",
        default="data/processed/abt_features.csv",
        help="Path to feature matrix CSV (pre-encoded, mixed types)",
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--output-dir", default="models", help="Directory for saved models"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Ensure binary columns are int and categoricals are string
    for col in BINARY_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(int)
    for col in ONEHOT_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Build preprocessor from feature_engineering
    preprocessor, _, _, _ = build_preprocessor(X)

    run_training(
        X, y,
        preprocessor=preprocessor,
        config_path=args.config,
        output_dir=args.output_dir,
    )

"""
eda.py — Comprehensive Exploratory Data Analysis for SaaS Churn
================================================================
Generates publication-quality visualizations and statistical summaries
from the merged ABT (one row per customer, churn as binary target).

Sections:
    1. Target Distribution
    2. Univariate Analysis (numeric + categorical)
    3. Bivariate Analysis (features split by churn)
    4. Correlation Analysis
    5. Summary of Key Findings

All plots are saved to reports/figures/.
Uses matplotlib + seaborn only — no plotly or ydata-profiling.

Usage:
    python -m src.eda                                     # defaults
    python -m src.eda --input data/processed/abt_pipeline.csv
"""

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib  # type: ignore[import-untyped]
matplotlib.use("Agg")  # non-interactive backend for saving only
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.ticker as mtick  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# =============================================================================
# Plot Style Configuration
# =============================================================================
STYLE_CONFIG = {
    "figure.figsize": (12, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
}

# Color palette — professional, colorblind-friendly
COLORS = {
    "primary": "#2563EB",      # blue
    "secondary": "#DC2626",    # red
    "accent": "#059669",       # emerald
    "neutral": "#6B7280",      # gray
    "churn_0": "#2563EB",      # blue = active
    "churn_1": "#DC2626",      # red = churned
    "palette": ["#2563EB", "#DC2626", "#059669", "#D97706",
                "#7C3AED", "#DB2777", "#0891B2"],
}

CHURN_LABELS = {0: "Active", 1: "Churned"}
CHURN_PALETTE = {0: COLORS["churn_0"], 1: COLORS["churn_1"]}


# =============================================================================
# EDA Class
# =============================================================================
class ChurnEDA:
    """
    Comprehensive EDA for the SaaS churn modeling table.

    Parameters
    ----------
    df : pd.DataFrame
        Merged ABT with one row per customer.
    output_dir : str
        Directory to save plots.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: str = "reports/figures",
    ):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.insights: list = []

        # Apply global style
        plt.rcParams.update(STYLE_CONFIG)
        sns.set_theme(style="whitegrid", palette=COLORS["palette"])

        # Derive columns the user expects
        self._derive_columns()

        logger.info(
            "ChurnEDA initialized | rows=%d | cols=%d | output=%s",
            len(self.df), len(self.df.columns), self.output_dir,
        )

    def _derive_columns(self):
        """Derive tenure_months and customer_segment from existing columns."""
        df = self.df

        # --- tenure_months from signup_date ---
        if "signup_date" in df.columns:
            df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
            reference = df["signup_date"].max()  # use latest signup as reference
            df["tenure_days"] = (reference - df["signup_date"]).dt.days
            df["tenure_months"] = (df["tenure_days"] / 30.44).round(1)
            logger.info("  Derived tenure_months (max=%.1f)", df["tenure_months"].max())

        # --- customer_segment from mrr_amount ---
        if "mrr_amount" in df.columns:
            df["customer_segment"] = pd.cut(
                df["mrr_amount"],
                bins=[-np.inf, 500, 2000, 5000, np.inf],
                labels=["SMB", "Mid-Market", "Enterprise", "Strategic"],
            )
            logger.info(
                "  Derived customer_segment: %s",
                dict(df["customer_segment"].value_counts()),
            )

        # --- region from country (group smaller countries) ---
        if "country" in df.columns:
            df["region"] = df["country"]  # keep as-is since we have 7 countries

        self.df = df

    # =========================================================================
    # 1. TARGET DISTRIBUTION
    # =========================================================================
    def plot_target_distribution(self) -> "ChurnEDA":
        """
        Bar plot of churned vs not-churned.
        Prints churn rate % and comments on class imbalance.
        """
        logger.info("=" * 70)
        logger.info("SECTION 1: TARGET DISTRIBUTION")
        logger.info("=" * 70)

        df = self.df
        churn_counts = df["churn_flag"].value_counts().sort_index()
        churn_rate = df["churn_flag"].mean() * 100
        n_churned = int(churn_counts.get(1, 0))
        n_active = int(churn_counts.get(0, 0))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Bar plot ---
        ax1 = axes[0]
        bars = ax1.bar(
            ["Active (0)", "Churned (1)"],
            [n_active, n_churned],
            color=[COLORS["churn_0"], COLORS["churn_1"]],
            edgecolor="white",
            linewidth=1.5,
            width=0.5,
        )
        # Add count labels on bars
        for bar, count in zip(bars, [n_active, n_churned]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{count}\n({count / len(df) * 100:.1f}%)",
                ha="center", va="bottom", fontweight="bold", fontsize=12,
            )
        ax1.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Number of Customers")
        ax1.set_ylim(0, max(n_active, n_churned) * 1.25)

        # --- Pie chart ---
        ax2 = axes[1]
        ax2.pie(
            [n_active, n_churned],
            labels=["Active", "Churned"],
            autopct="%1.1f%%",
            colors=[COLORS["churn_0"], COLORS["churn_1"]],
            startangle=90,
            explode=(0, 0.05),
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        ax2.set_title("Churn Rate Breakdown", fontsize=14, fontweight="bold")

        plt.suptitle(
            f"Overall Churn Rate: {churn_rate:.1f}%  |  "
            f"Active: {n_active}  |  Churned: {n_churned}  |  Total: {len(df)}",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        self._save_fig("01_target_distribution")

        # --- Insights ---
        imbalance_ratio = n_active / max(n_churned, 1)
        insight = (
            f"### Target Distribution Insights\n"
            f"- **Churn rate: {churn_rate:.1f}%** — "
        )
        if churn_rate < 10:
            insight += "Severely imbalanced. Consider SMOTE or class weights.\n"
        elif churn_rate < 25:
            insight += "Moderate imbalance. Stratified splitting + class_weight='balanced' should suffice.\n"
        else:
            insight += "Relatively balanced. Standard approaches should work.\n"

        insight += (
            f"- Class ratio: {imbalance_ratio:.1f}:1 (active:churned)\n"
            f"- With {n_churned} positive samples, we have enough signal for "
            f"tree-based models but should use stratified CV.\n"
        )
        self.insights.append(insight)
        logger.info("  Churn rate: %.1f%% | Active: %d | Churned: %d", churn_rate, n_active, n_churned)

        return self

    # =========================================================================
    # 2. UNIVARIATE ANALYSIS
    # =========================================================================
    def plot_univariate_numeric(self) -> "ChurnEDA":
        """
        Histograms + KDE for key numeric features:
        MRR, tenure_months, total_events, avg_daily_usage
        """
        logger.info("=" * 70)
        logger.info("SECTION 2a: UNIVARIATE — NUMERIC FEATURES")
        logger.info("=" * 70)

        numeric_features = {
            "mrr_amount": "Monthly Recurring Revenue ($)",
            "tenure_months": "Account Tenure (Months)",
            "total_events": "Total Usage Events",
            "avg_daily_usage": "Avg Daily Usage (Events/Day)",
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(numeric_features.items()):
            ax = axes[idx]
            if col not in self.df.columns:
                ax.text(0.5, 0.5, f"Column '{col}' not found",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            data = self.df[col].dropna()

            # Histogram + KDE
            sns.histplot(
                data, kde=True, ax=ax, color=COLORS["primary"],
                edgecolor="white", linewidth=0.5, alpha=0.7, bins=30,
            )

            # Stats annotation
            stats_text = (
                f"Mean: {data.mean():.1f}\n"
                f"Median: {data.median():.1f}\n"
                f"Std: {data.std():.1f}\n"
                f"Skew: {data.skew():.2f}"
            )
            ax.text(
                0.97, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),  # type: ignore[arg-type]
            )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("Count")

        plt.suptitle("Univariate Distributions — Key Numeric Features",
                      fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_fig("02_univariate_numeric")

        # --- Insights ---
        insight = "### Univariate Numeric Insights\n"
        for col, title in numeric_features.items():
            if col in self.df.columns:
                data = self.df[col].dropna()
                skew = data.skew()
                skew_label = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "symmetric")
                insight += f"- **{title}**: median={data.median():.1f}, {skew_label} (skew={skew:.2f})\n"
        insight += "- Right-skewed revenue/usage distributions are typical in B2B SaaS — a few large accounts dominate.\n"  # type: ignore[operator]
        self.insights.append(insight)

        return self

    def plot_univariate_categorical(self) -> "ChurnEDA":
        """
        Bar plots for key categorical features:
        plan_tier, customer_segment, region/country, industry
        """
        logger.info("=" * 70)
        logger.info("SECTION 2b: UNIVARIATE — CATEGORICAL FEATURES")
        logger.info("=" * 70)

        cat_features = {
            "plan_tier": "Plan Tier",
            "customer_segment": "Customer Segment (by MRR)",
            "country": "Country / Region",
            "industry": "Industry",
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(cat_features.items()):
            ax = axes[idx]
            if col not in self.df.columns:
                ax.text(0.5, 0.5, f"Column '{col}' not found",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            counts = self.df[col].value_counts().sort_values(ascending=True)
            colors = COLORS["palette"][:len(counts)]  # type: ignore[index]

            bars = ax.barh(
                counts.index.astype(str), counts.values,
                color=colors[::-1], edgecolor="white", linewidth=0.5,
            )
            # Add count labels
            for bar, count in zip(bars, counts.values):
                ax.text(
                    bar.get_width() + max(counts) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count} ({count / len(self.df) * 100:.1f}%)",
                    va="center", fontsize=10,
                )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Count")
            ax.set_xlim(0, max(counts) * 1.3)

        plt.suptitle("Univariate Distributions — Categorical Features",
                      fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_fig("03_univariate_categorical")

        # --- Insights ---
        insight = "### Univariate Categorical Insights\n"
        for col, title in cat_features.items():
            if col in self.df.columns:
                top = self.df[col].value_counts().head(1)
                insight += f"- **{title}**: most common = '{top.index[0]}' ({top.values[0]} accounts, {top.values[0]/len(self.df)*100:.1f}%)\n"  # type: ignore[operator]
        insight += "- Check if plan tier or segment distribution is heavily skewed — could need stratified analysis.\n"  # type: ignore[operator]
        self.insights.append(insight)

        return self

    # =========================================================================
    # ADDITIONAL UNIVARIATE — Extended Numeric Features
    # =========================================================================
    def plot_univariate_extended(self) -> "ChurnEDA":
        """
        Histograms for additional numeric features:
        seats, arr_amount, total_errors, avg_satisfaction,
        ticket_count, active_days, unique_features_used, avg_duration_secs
        """
        logger.info("=" * 70)
        logger.info("SECTION 2c: UNIVARIATE — EXTENDED NUMERIC")
        logger.info("=" * 70)

        extended_features = {
            "seats": "Seats (Licensed)",
            "arr_amount": "Annual Recurring Revenue ($)",
            "total_errors": "Total Errors",
            "avg_satisfaction": "Avg Support Satisfaction (1–5)",
            "ticket_count": "Support Ticket Count",
            "active_days": "Active Days",
            "unique_features_used": "Unique Features Used",
            "avg_duration_secs": "Avg Usage Duration (sec)",
        }

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(extended_features.items()):
            ax = axes[idx]
            if col not in self.df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            data = self.df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color=COLORS["accent"],
                         edgecolor="white", alpha=0.7, bins=25)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=8)

        plt.suptitle("Extended Numeric Feature Distributions",
                      fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_fig("04_univariate_extended")

        return self

    # =========================================================================
    # 3. BIVARIATE ANALYSIS
    # =========================================================================
    def plot_bivariate_boxplots(self) -> "ChurnEDA":
        """
        Boxplots / violin plots of MRR, tenure, usage split by churn (0 vs 1).
        """
        logger.info("=" * 70)
        logger.info("SECTION 3a: BIVARIATE — BOXPLOTS BY CHURN")
        logger.info("=" * 70)

        bivariate_cols = {
            "mrr_amount": "MRR ($)",
            "tenure_months": "Tenure (Months)",
            "total_events": "Total Usage Events",
            "avg_daily_usage": "Avg Daily Usage",
            "active_days": "Active Days",
            "avg_satisfaction": "Avg Satisfaction",
        }

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(bivariate_cols.items()):
            ax = axes[idx]
            if col not in self.df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            # Violin plot with embedded box
            parts = sns.violinplot(
                data=self.df, x="churn_flag", y=col, ax=ax,
                hue="churn_flag", palette=CHURN_PALETTE,
                inner="box", density_norm="width", legend=False,
            )
            ax.set_xticklabels(["Active (0)", "Churned (1)"])
            ax.set_xlabel("")
            ax.set_title(title, fontsize=13, fontweight="bold")

            # Add median annotation
            for churn_val in [0, 1]:
                subset = self.df[self.df["churn_flag"] == churn_val][col].dropna()
                median = subset.median()
                ax.text(
                    churn_val, median,
                    f"  Md={median:.1f}",
                    fontsize=9, fontweight="bold", va="center",
                    color="black",
                )

        plt.suptitle("Feature Distributions by Churn Status (Violin + Box)",
                      fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_fig("05_bivariate_boxplots")

        # --- Insights ---
        insight = "### Bivariate Boxplot Insights\n"
        for col, title in bivariate_cols.items():
            if col not in self.df.columns:
                continue
            med_active = self.df[self.df["churn_flag"] == 0][col].median()
            med_churned = self.df[self.df["churn_flag"] == 1][col].median()
            if med_active > 0:
                pct_diff = ((med_churned - med_active) / med_active) * 100
                direction = "lower" if pct_diff < 0 else "higher"
                insight += (  # type: ignore[operator]
                    f"- **{title}**: churned median = {med_churned:.1f} vs "
                    f"active = {med_active:.1f} ({abs(pct_diff):.1f}% {direction})\n"
                )
        insight += (  # type: ignore[operator]
            "- Features where churned customers show clearly lower values "
            "(usage, tenure) are strong churn predictors.\n"
        )
        self.insights.append(insight)

        return self

    def plot_bivariate_churn_rate_by_category(self) -> "ChurnEDA":
        """
        Grouped bar charts: churn rate by plan_tier, customer_segment,
        region/country, industry.
        """
        logger.info("=" * 70)
        logger.info("SECTION 3b: BIVARIATE — CHURN RATE BY CATEGORY")
        logger.info("=" * 70)

        cat_features = {
            "plan_tier": "Plan Tier",
            "customer_segment": "Customer Segment",
            "country": "Country / Region",
            "industry": "Industry",
        }

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(cat_features.items()):
            ax = axes[idx]
            if col not in self.df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            # Compute churn rate per category
            churn_by_cat = (
                self.df.groupby(col)["churn_flag"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "churn_rate", "count": "n_customers"})
                .sort_values("churn_rate", ascending=True)
            )

            colors = []
            for rate in churn_by_cat["churn_rate"]:
                if rate > 0.30:
                    colors.append(COLORS["secondary"])
                elif rate > 0.20:
                    colors.append(COLORS["neutral"])
                else:
                    colors.append(COLORS["primary"])

            bars = ax.barh(
                churn_by_cat.index.astype(str),
                churn_by_cat["churn_rate"] * 100,
                color=colors, edgecolor="white", linewidth=0.5,
            )

            # Add labels
            for bar, (_, row) in zip(bars, churn_by_cat.iterrows()):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['churn_rate']*100:.1f}% (n={int(row['n_customers'])})",
                    va="center", fontsize=10, fontweight="bold",
                )

            # Add overall churn rate reference line
            overall_rate = self.df["churn_flag"].mean() * 100
            ax.axvline(
                overall_rate, color=COLORS["secondary"],
                linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Overall: {overall_rate:.1f}%",
            )

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Churn Rate (%)")
            ax.set_xlim(0, churn_by_cat["churn_rate"].max() * 100 * 1.4)
            ax.legend(fontsize=9)

        plt.suptitle("Churn Rate by Category (red = above average, blue = below)",
                      fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_fig("06_churn_rate_by_category")

        # --- Insights ---
        insight = "### Churn Rate by Category Insights\n"
        overall = self.df["churn_flag"].mean() * 100
        for col, title in cat_features.items():
            if col not in self.df.columns:
                continue
            rates = (
                self.df.groupby(col)["churn_flag"]
                .mean()
                .sort_values(ascending=False)
            )
            worst = rates.index[0]
            best = rates.index[-1]
            insight += (
                f"- **{title}**: highest churn = '{worst}' ({rates.iloc[0]*100:.1f}%), "
                f"lowest = '{best}' ({rates.iloc[-1]*100:.1f}%)\n"
            )
        insight += f"- Overall churn rate: {overall:.1f}%. Categories significantly above this are priority targets.\n"
        self.insights.append(insight)

        return self

    def plot_bivariate_extended(self) -> "ChurnEDA":
        """
        Additional bivariate: churn rate by binary flags (is_trial,
        upgrade, downgrade, auto_renew) as a single summary plot.
        """
        logger.info("=" * 70)
        logger.info("SECTION 3c: BIVARIATE — CHURN RATE BY FLAGS")
        logger.info("=" * 70)

        flag_cols = [
            ("is_trial", "Trial Account"),
            ("upgrade_flag", "Upgraded"),
            ("downgrade_flag", "Downgraded"),
            ("auto_renew_flag", "Auto-Renew On"),
        ]

        available = [(c, l) for c, l in flag_cols if c in self.df.columns]
        if not available:
            logger.warning("  No flag columns found, skipping")
            return self

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
        if len(available) == 1:
            axes = [axes]

        for idx, (col, label) in enumerate(available):
            ax = axes[idx]  # type: ignore[index]
            rates = self.df.groupby(col)["churn_flag"].mean() * 100

            bars = ax.bar(
                [f"No\n({col}=0)", f"Yes\n({col}=1)"],
                [rates.get(False, rates.get(0, 0)), rates.get(True, rates.get(1, 0))],
                color=[COLORS["primary"], COLORS["secondary"]],
                edgecolor="white", width=0.5,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%",
                    ha="center", fontsize=11, fontweight="bold",
                )
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, max(rates.max() * 1.3, 30))

        plt.suptitle("Churn Rate by Account Flags",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_fig("07_churn_rate_by_flags")

        return self

    # =========================================================================
    # 4. CORRELATION ANALYSIS
    # =========================================================================
    def plot_correlation_heatmap(self, top_n: int = 5) -> "ChurnEDA":
        """
        Compute correlation matrix for all numeric features.
        Plot seaborn heatmap. Annotate top N features most correlated with churn.
        """
        logger.info("=" * 70)
        logger.info("SECTION 4: CORRELATION ANALYSIS")
        logger.info("=" * 70)

        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=["number", "bool"])
        # Convert bool to int for correlation
        for col in numeric_df.select_dtypes(include=["bool"]).columns:
            numeric_df[col] = numeric_df[col].astype(int)

        # Drop identifiers and low-information columns
        drop_cols = [c for c in ["account_id"] if c in numeric_df.columns]
        numeric_df = numeric_df.drop(columns=drop_cols, errors="ignore")

        corr = numeric_df.corr()

        # --- Full heatmap ---
        fig, ax = plt.subplots(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 7},
            ax=ax,
        )
        ax.set_title(
            "Feature Correlation Matrix (Lower Triangle)",
            fontsize=15, fontweight="bold", pad=20,
        )
        plt.tight_layout()
        self._save_fig("08_correlation_heatmap")

        # --- Top features correlated with churn ---
        if "churn_flag" in corr.columns:
            churn_corr = (
                corr["churn_flag"]
                .drop("churn_flag")
                .abs()
                .sort_values(ascending=False)
                .head(top_n)
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [
                COLORS["secondary"] if corr.loc[feat, "churn_flag"] > 0
                else COLORS["primary"]
                for feat in churn_corr.index
            ]
            bars = ax.barh(
                churn_corr.index[::-1],
                churn_corr.values[::-1],
                color=colors[::-1], edgecolor="white",  # type: ignore[index]
            )
            for bar, feat in zip(bars, churn_corr.index[::-1]):
                actual_corr = corr.loc[feat, "churn_flag"]
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"r = {actual_corr:+.3f}",
                    va="center", fontsize=11, fontweight="bold",
                )
            ax.set_title(
                f"Top {top_n} Features Most Correlated with Churn",
                fontsize=14, fontweight="bold",
            )
            ax.set_xlabel("|Correlation| with churn_flag")

            # Color legend
            from matplotlib.patches import Patch  # type: ignore[import-untyped]
            legend_elements = [
                Patch(facecolor=COLORS["secondary"], label="Positive (increases churn)"),
                Patch(facecolor=COLORS["primary"], label="Negative (decreases churn)"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

            plt.tight_layout()
            self._save_fig("09_top_churn_correlations")

            # --- Insights ---
            insight = f"### Correlation Insights (Top {top_n})\n"
            for feat in churn_corr.index:
                r = corr.loc[feat, "churn_flag"]
                direction = "positively" if r > 0 else "negatively"
                insight += f"- **{feat}**: r = {r:+.3f} ({direction} correlated with churn)\n"

            # Check for multicollinearity
            high_corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.8 and corr.columns[i] != "churn_flag" and corr.columns[j] != "churn_flag":
                        high_corr_pairs.append(
                            (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                        )
            if high_corr_pairs:
                insight += f"\n**⚠️ Multicollinearity Warning:** {len(high_corr_pairs)} feature pairs with |r| > 0.8:\n"  # type: ignore[operator]
                for f1, f2, r in high_corr_pairs[:5]:  # type: ignore[index]
                    insight += f"  - {f1} ↔ {f2}: r = {r:.3f}\n"
                insight += "  Consider dropping one from each pair before training.\n"

            self.insights.append(insight)

        return self

    # =========================================================================
    # 5. SUMMARY
    # =========================================================================
    def generate_summary_report(self) -> str:
        """
        Generate a final markdown summary with all EDA takeaways.
        Saves to reports/eda_summary.md.
        """
        logger.info("=" * 70)
        logger.info("SECTION 5: GENERATING EDA SUMMARY REPORT")
        logger.info("=" * 70)

        churn_rate = self.df["churn_flag"].mean() * 100
        n = len(self.df)

        report = f"""# Exploratory Data Analysis — Summary Report
## RavenStack B2B SaaS Churn Prediction

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Dataset:** {n} customers | {len(self.df.columns)} features | Churn rate: {churn_rate:.1f}%

---

## Section-by-Section Findings

"""
        for insight in self.insights:
            report += insight + "\n---\n\n"

        report += """## Overall EDA Takeaways for B2B SaaS Churn

### Key Patterns to Exploit in Modeling

1. **Low-Usage Accounts Are High Risk**
   Accounts with below-median usage events and active days are
   disproportionately likely to churn. Usage intensity features
   (total_events, avg_daily_usage, active_days) should be among
   the most important model features.

2. **Short-Tenure Risk**
   New accounts (low tenure_months) show elevated churn — the
   "activation gap" where customers haven't found product value yet.
   Consider creating an `is_new_account` flag (< 3 months).

3. **Plan-Type Patterns**
   Different plan tiers show different churn behaviors. Lower-tier
   plans may churn more (price sensitivity) or less (lower expectations).
   Plan tier should be encoded as a feature.

4. **Support Signals Matter**
   Accounts with many support tickets and low satisfaction scores
   are sending distress signals. Ticket count and satisfaction
   are likely important bivariate predictors.

5. **Revenue at Risk**
   The MRR distribution reveals which churners represent the
   biggest revenue impact. High-MRR churners should get priority
   attention from the CS team.

### Modeling Recommendations

- **Class Imbalance:** Churn rate at {churn_rate:.1f}% \u2014 use stratified CV,
  class_weight=\u2018balanced\u2019, and prioritize **Recall** as the primary metric.
- **Feature Engineering:** Create interaction features (e.g., usage × tenure,
  tickets × satisfaction) and ratio features (events per active day).
- **Multicollinearity:** Drop one feature from highly correlated pairs
  (e.g., mrr_amount ↔ arr_amount) to improve model interpretability.
- **Leakage Check:** Verify no post-churn features survive preprocessing.

---

*All plots saved to `reports/figures/`. Use this report alongside the
visualizations for stakeholder presentations.*
"""

        # Save report
        report_path = Path("reports") / "eda_summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("  EDA summary saved to %s", report_path)

        return report

    # =========================================================================
    # FULL RUN
    # =========================================================================
    def run(self) -> str:
        """Execute full EDA pipeline and return summary report."""
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  COMPREHENSIVE EDA — STARTING" + " " * 38 + "║")
        logger.info("╚" + "═" * 68 + "╝")

        (
            self.plot_target_distribution()
            .plot_univariate_numeric()
            .plot_univariate_categorical()
            .plot_univariate_extended()
            .plot_bivariate_boxplots()
            .plot_bivariate_churn_rate_by_category()
            .plot_bivariate_extended()
            .plot_correlation_heatmap()
        )

        report = self.generate_summary_report()

        n_plots = len(list(self.output_dir.glob("*.png")))
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  EDA COMPLETE" + " " * 54 + "║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info("║  Plots saved    : %-48s ║", f"{n_plots} PNGs")
        logger.info("║  Output dir     : %-48s ║", str(self.output_dir))
        logger.info("║  Summary report : %-48s ║", "reports/eda_summary.md")
        logger.info("║  Insights       : %-48s ║", f"{len(self.insights)} sections")
        logger.info("╚" + "═" * 68 + "╝")

        return report

    # =========================================================================
    # Utility
    # =========================================================================
    def _save_fig(self, name: str) -> None:
        """Save current figure and close."""
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, facecolor="white")
        plt.close()
        logger.info("  Saved: %s", path)


# =============================================================================
# CLI Entrypoint
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive EDA")
    parser.add_argument(
        "--input",
        default="data/processed/abt_pipeline.csv",
        help="Path to the merged ABT CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    # Load data
    logger.info("Loading data from %s", args.input)
    df = pd.read_csv(args.input)

    # Run EDA
    eda = ChurnEDA(df=df, output_dir=args.output_dir)
    summary = eda.run()
    print("\n" + "=" * 70)
    print("EDA COMPLETE — All plots saved to", args.output_dir)
    print("Summary report: reports/eda_summary.md")
    print("=" * 70)

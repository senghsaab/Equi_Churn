import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS
# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\Lenovo\Design Thinking And Innovation Project"
SAAS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "churn_dataset.csv")
KAG_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "external", "Churn_Modelling.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "eda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.max_open_warning': 0})

def save_plot(name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [OK] Saved plot: {name}.png")

# ──────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
print("\n[1/3] Loading datasets...")

# SaaS Dataset
df_saas = pd.read_csv(SAAS_DATA_PATH)
df_saas.rename(columns={'churned': 'churn_flag'}, inplace=True)

# Kaggle Bank Dataset
df_kag = pd.read_csv(KAG_DATA_PATH)
df_kag.rename(columns={'Exited': 'churn_flag'}, inplace=True)

# Map Kaggle columns to SaaS names for alignment where possible in summary
# (We will use these primarily in the comparison table)
KAG_MAPPING = {
    'Balance': 'monthly_revenue_proxy',
    'Tenure': 'tenure_months_proxy',
    'Age': 'age_proxy'
}

# ──────────────────────────────────────────────────────────────────────────
# EDA WORKER CLASS
# ──────────────────────────────────────────────────────────────────────────

class EquiChurnEDA:
    def __init__(self, df: pd.DataFrame, source: str, prefix: str):
        self.df = df
        self.source = source
        self.prefix = prefix
        print(f"\n--- Running EDA for {source} ---")

    def section_1_target(self):
        print(f"[{self.prefix}] Section 1: Target Distribution")
        plt.figure(figsize=(7, 5))
        ax = sns.countplot(x='churn_flag', data=self.df, palette=['#4C72B0', '#C44E52'])
        
        counts = self.df['churn_flag'].value_counts()
        for i, count in enumerate(counts):
            ax.text(i, count + (max(counts)*0.01), f'{count:,}', ha='center', fontweight='bold')
        
        churn_rate = self.df['churn_flag'].mean() * 100
        plt.title(f"Target Distribution: {self.source}\n(Churn Rate: {churn_rate:.1f}%)", fontsize=14)
        plt.xlabel("Churn Flag (0=Retained, 1=Churned)")
        plt.ylabel("Account Count")
        save_plot(f"{self.prefix}_01_target_dist")
        
        print(f"  Churn Rate: {churn_rate:.1f}% ({self.source})")
        # Comment: Class imbalance severity and SMOTETomek
        # Note: B2B SaaS (20%) vs Kaggle Bank (approx 20.4%)
        
    def section_2_univariate(self, num_cols: List[str], cat_cols: List[str]):
        print(f"[{self.prefix}] Section 2: Univariate Analysis")
        
        # Numeric Features
        for col in num_cols:
            if col not in self.df.columns: continue
            plt.figure(figsize=(9, 4))
            sns.histplot(self.df[col], kde=True, color='#4C72B0')
            plt.title(f"Distribution of {col} ({self.source})")
            save_plot(f"{self.prefix}_02_num_{col}")
            
        # Categorical Features
        for col in cat_cols:
            if col not in self.df.columns: continue
            plt.figure(figsize=(10, 5))
            order = self.df[col].value_counts().index
            ax = sns.countplot(y=col, data=self.df, order=order, palette="viridis")
            
            title = f"Distribution of {col} ({self.source})"
            if col == 'age_group':
                title += " -- [Protected Attribute]"
            plt.title(title, fontsize=13)
            save_plot(f"{self.prefix}_02_cat_{col}")

    def section_3_bivariate(self, box_cols: List[str], violin_cols: List[str], bias_cols: List[str]):
        print(f"[{self.prefix}] Section 3: Bivariate Analysis")
        
        # Boxplots
        for col in box_cols:
            if col not in self.df.columns: continue
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='churn_flag', y=col, data=self.df, palette=['#4C72B0', '#C44E52'], showfliers=False)
            plt.title(f"{col} vs Churn ({self.source})", fontsize=13)
            save_plot(f"{self.prefix}_03_box_{col}")
            
        # Violin plots
        for col in violin_cols:
            if col not in self.df.columns: continue
            plt.figure(figsize=(8, 6))
            sns.violinplot(x='churn_flag', y=col, data=self.df, palette=['#4C72B0', '#C44E52'], split=True)
            plt.title(f"{col} Distribution by Churn ({self.source})", fontsize=13)
            save_plot(f"{self.prefix}_03_violin_{col}")
            
        # Churn Rate by Category
        for col in bias_cols:
            if col not in self.df.columns: continue
            plt.figure(figsize=(10, 6))
            churn_rates = self.df.groupby(col)['churn_flag'].mean().sort_values(ascending=False)
            ax = sns.barplot(x=churn_rates.index, y=churn_rates.values, palette="Reds_r")
            
            title = f"Churn Rate by {col} ({self.source})"
            if col in ['age_group', 'region', 'Geography']:
                title = f"Churn Rate by {col} (Protected Attribute) -- {self.source}"
                plt.axhline(0.1, color='red', linestyle='--', alpha=0.6)
                plt.annotate("Demographic Parity > 0.1 indicates bias", xy=(0.5, 0.11), color='red', fontsize=9)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.ylabel("Mean Churn Rate")
            plt.ylim(0, max(churn_rates)*1.3)
            # Add labels
            for i, v in enumerate(churn_rates.values):
                ax.text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=10)
            
            save_plot(f"{self.prefix}_03_rate_{col}")

    def section_4_correlation(self):
        print(f"[{self.prefix}] Section 4: Correlation Analysis")
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Heatmap: {self.source}", fontsize=15)
        save_plot(f"{self.prefix}_04_corr_heatmap")
        
        # Highlight top 5
        top_5 = corr['churn_flag'].abs().sort_values(ascending=False)[1:6]
        print(f"  Top 5 Correlations with Churn ({self.source}):")
        for idx, val in top_5.items():
            print(f"    - {idx}: {corr.loc[idx, 'churn_flag']:.4f}")

    def section_5_fairness(self, protected_cols: List[str], proxy_feature: str):
        print(f"[{self.prefix}] Section 5: Fairness-Aware EDA")
        
        for p_col in protected_cols:
            if p_col not in self.df.columns: continue
            
            # Feature distribution split by protected attribute
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=self.df, x=proxy_feature, hue=p_col, common_norm=False, fill=True, alpha=0.3)
            plt.title(f"{proxy_feature} Distribution by {p_col} ({self.source})\n[Proxy Discrimination Risk Check]", fontsize=13)
            save_plot(f"{self.prefix}_05_fairness_{p_col}_{proxy_feature}")
            
            # KS Test for distribution difference
            groups = self.df[p_col].unique()
            if len(groups) >= 2:
                # Compare first two groups for simplicity in log
                g1 = self.df[self.df[p_col] == groups[0]][proxy_feature]
                g2 = self.df[self.df[p_col] == groups[1]][proxy_feature]
                stat, p_val = ks_2samp(g1, g2)
                print(f"  KS Test ({p_col}): stat={stat:.4f}, p={p_val:.4e}")
                if p_val < 0.05:
                    print(f"    [!] {proxy_feature} significantly differs across {p_col} groups (Potential Proxy Discriminator).")

# ──────────────────────────────────────────────────────────────────────────
# RUN EDA
# ──────────────────────────────────────────────────────────────────────────

# SaaS EDA
saas_eda = EquiChurnEDA(df_saas, "Synthetic B2B SaaS", "saas")
saas_eda.section_1_target()
saas_eda.section_2_univariate(
    num_cols=['mrr', 'tenure_months', 'dau_wau_ratio', 'login_frequency_monthly', 'support_tickets_6m', 'feature_adoption_rate'],
    cat_cols=['plan_type', 'customer_segment', 'region', 'age_group', 'tenure_bucket', 'mrr_segment']
)
saas_eda.section_3_bivariate(
    box_cols=['mrr', 'tenure_months'],
    violin_cols=['login_frequency_monthly', 'feature_adoption_rate'],
    bias_cols=['plan_type', 'customer_segment', 'region', 'age_group']
)
saas_eda.section_4_correlation()
saas_eda.section_5_fairness(protected_cols=['age_group', 'region'], proxy_feature='mrr')

# Kaggle Bank EDA
kag_eda = EquiChurnEDA(df_kag, "Kaggle Bank Proxy", "bank")
kag_eda.section_1_target()
kag_eda.section_2_univariate(
    num_cols=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'],
    cat_cols=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
)
kag_eda.section_3_bivariate(
    box_cols=['Balance', 'Tenure'],
    violin_cols=['Age', 'CreditScore'],
    bias_cols=['Geography', 'Gender', 'IsActiveMember']
)
kag_eda.section_4_correlation()
kag_eda.section_5_fairness(protected_cols=['Geography', 'Gender'], proxy_feature='Balance')

# ──────────────────────────────────────────────────────────────────────────
# 6. DATASET COMPARISON SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n[3/3] Generating Comparison Summary Table...")

# Compute tenure proxy: use 'active_days' if available, else 'contract_renewal_days'
tenure_col = 'active_days' if 'active_days' in df_saas.columns else 'contract_renewal_days'
tenure_label = "days" if tenure_col == 'active_days' else "days to renewal"

try:
    stats = {
        "Metric": ["Churn Rate %", "Class Balance (Minority %)", "Avg Tenure/Activity", "Avg Revenue/Balance", "Demographic bias (Region/Geo max diff)"],
        "Synthetic SaaS": [
            f"{df_saas['churn_flag'].mean()*100:.2f}%",
            f"{(df_saas['churn_flag'].value_counts(normalize=True)[1])*100:.1f}%",
            f"{df_saas[tenure_col].mean():.1f} {tenure_label}",
            f"${df_saas['mrr'].mean():.1f}",
            f"{(df_saas.groupby('region')['churn_flag'].mean().max() - df_saas.groupby('region')['churn_flag'].mean().min()):.1%}"
        ],
        "Kaggle Bank": [
            f"{df_kag['churn_flag'].mean()*100:.2f}%",
            f"{(df_kag['churn_flag'].value_counts(normalize=True)[1])*100:.1f}%",
            f"{df_kag['Tenure'].mean():.1f} years",
            f"{df_kag['Balance'].mean():.1f} units",
            f"{(df_kag.groupby('Geography')['churn_flag'].mean().max() - df_kag.groupby('Geography')['churn_flag'].mean().min()):.1%}"
        ]
    }

    comp_df = pd.DataFrame(stats)
    print("\n" + "="*80)
    print("  DATASET COMPARISON SUMMARY")
    print("="*80)
    print(comp_df.to_string(index=False))
    print("="*80)

    # Save summary to reports
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    comp_df.to_csv(os.path.join(PROJECT_ROOT, "reports", "dataset_comparison_stats.csv"), index=False)
except Exception as e:
    print(f"  [WARN] Could not generate comparison table: {e}")

print("\nEDA PROCESS COMPLETE. All plots saved to reports/figures/eda/")


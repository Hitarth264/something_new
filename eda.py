"""
Exploratory Data Analysis for Customer Churn Dataset.
Generates summary stats and saves visualizations to outputs/.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/customer_churn.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", df["Churn"].value_counts(normalize=True))

# --- Target distribution ---
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn Class Distribution")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/churn_distribution.png", dpi=150)
plt.close()

# --- Numeric feature distributions by churn ---
numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                 "Payment Delay", "Total Spend", "Last Interaction"]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x="Churn", y=col, ax=axes[i], palette=["#4C72B0", "#DD8452"])
    axes[i].set_title(col)
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/numeric_features_by_churn.png", dpi=150)
plt.close()

# --- Categorical features by churn ---
cat_cols = ["Gender", "Subscription Type", "Contract Length"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=axes[i], palette="viridis")
    axes[i].set_title(f"Churn Rate by {col}")
    axes[i].set_ylabel("Churn Rate")
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/categorical_churn_rates.png", dpi=150)
plt.close()

# --- Correlation heatmap ---
plt.figure(figsize=(8, 6))
corr = df[numeric_cols + ["Churn"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

print(f"\nEDA plots saved to {OUT_DIR}/")

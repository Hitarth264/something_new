"""
Train and evaluate churn prediction models.
Compares Logistic Regression, Random Forest, and XGBoost.
Saves the best model + evaluation artifacts to models/ and outputs/.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

RANDOM_STATE = 42
OUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("data/customer_churn.csv")
df = df.drop(columns=["CustomerID"])  # identifier, not predictive

X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                     "Payment Delay", "Total Spend", "Last Interaction"]
categorical_features = ["Gender", "Subscription Type", "Contract Length"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ---------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------
# 3. Define candidate models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
    ),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics
    fitted_pipelines[name] = pipe

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# ---------------------------------------------------------------
# 4. Compare models
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv")
print("\nModel comparison:\n", results_df)

best_model_name = results_df.index[0]
best_pipe = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------
# 5. Save best model
# ---------------------------------------------------------------
joblib.dump(best_pipe, f"{MODEL_DIR}/churn_model.joblib")
with open(f"{MODEL_DIR}/model_info.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "metrics": results[best_model_name],
        "features_numeric": numeric_features,
        "features_categorical": categorical_features,
    }, f, indent=2)

# ---------------------------------------------------------------
# 6. Visualizations for the best model
# ---------------------------------------------------------------
y_pred_best = best_pipe.predict(X_test)
y_proba_best = best_pipe.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ROC curves for all models
plt.figure(figsize=(6, 5))
for name, pipe in fitted_pipelines.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
plt.close()

# Feature importance (if available)
model_step = best_pipe.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    feature_names = (
        numeric_features +
        list(best_pipe.named_steps["preprocess"]
             .named_transformers_["cat"]
             .get_feature_names_out(categorical_features))
    )
    importances = pd.Series(model_step.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Top Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

print(f"\nAll outputs saved to {OUT_DIR}/ and model saved to {MODEL_DIR}/")

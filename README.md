# Customer Churn Prediction

A machine learning pipeline that predicts customer churn from account and usage data. The project covers exploratory data analysis, preprocessing, training/comparison of three classifiers, and a ready-to-use inference script.

## Dataset

- **Source:** Customer churn dataset (Kaggle-style synthetic data), 64,374 records, 12 columns.
- **Target:** `Churn` (1 = customer churned, 0 = retained), roughly balanced (~47% / 53%).
- **Features:**
  - Numeric: `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`
  - Categorical: `Gender`, `Subscription Type` (Basic/Standard/Premium), `Contract Length` (Monthly/Quarterly/Annual)
  - Dropped: `CustomerID` (identifier only, not predictive)
- No missing values or duplicate records.

## Project Structure

```
churn_project/
├── data/
│   ├── customer_churn.csv          # raw dataset
│   └── sample_new_customers.csv    # example input for predict.py
├── src/
│   ├── eda.py                      # exploratory analysis, saves plots to outputs/
│   ├── train_model.py              # trains/compares models, saves best one to models/
│   └── predict.py                  # CLI script to score new customers
├── models/
│   ├── churn_model.joblib          # trained pipeline (preprocessing + model)
│   └── model_info.json             # metadata: best model, metrics, feature lists
├── outputs/                        # generated plots and metrics (created on run)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Exploratory analysis** (writes plots to `outputs/`):
```bash
python src/eda.py
```

**2. Train and compare models** (writes the best model to `models/` and metrics/plots to `outputs/`):
```bash
python src/train_model.py
```

**3. Predict on new data:**
```bash
python src/predict.py --input data/sample_new_customers.csv --output outputs/predictions.csv
```
Input CSV needs the same columns as the training data (minus `Churn`); `CustomerID` is optional and passed through to the output.

## Modeling Approach

Three classifiers were trained inside a single `scikit-learn` pipeline (`ColumnTransformer` for preprocessing → estimator), using an 80/20 stratified train/test split:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.827 | 0.814 | 0.823 | 0.819 | 0.903 |
| Random Forest | 0.998 | 0.999 | 0.997 | 0.998 | 1.000 |
| XGBoost | 0.9999 | 0.9998 | 1.000 | 0.9999 | 1.000 |

XGBoost was selected as the saved model based on ROC AUC.

Preprocessing: numeric features are standardized (`StandardScaler`); categorical features are one-hot encoded (`OneHotEncoder`, dropping the first category to avoid collinearity).

### Important caveat on the near-perfect scores

The Random Forest and XGBoost results above (99.8–99.99%) are unusually high for a churn problem and are worth treating with skepticism rather than taking at face value. Investigation of this dataset shows `Churn` is almost fully determined by a handful of features — primarily `Payment Delay` and `Support Calls`, with secondary contributions from `Gender`, `Usage Frequency`, `Tenure`, and `Contract Length` — through what looks like a rule-based or formulaic generation process rather than organic customer behavior. A shallow (depth-4) decision tree alone reaches 94% accuracy on this data, which would not happen with real-world churn data this clean. There are no duplicate rows or literal target leakage (e.g., a post-outcome column), but the data is essentially "too learnable" for tree ensembles.

Logistic Regression's 90.3% AUC is the more realistic reference point here, since it can't fit the sharp, rule-like decision boundaries that trees pick up. **If you plan to apply this pipeline to real production data, re-validate performance on that data** — expect tree-based metrics to drop substantially outside this synthetic dataset, and treat XGBoost's feature importances (dominated by `Payment Delay`) as a property of this dataset's generation process rather than a universal truth about churn drivers.

## Key Findings (EDA)

- Customers with **higher payment delays and more support calls** churn at a much higher rate.
- **Monthly contracts** churn more than Quarterly or Annual contracts.
- **Female customers** churn at a higher rate than male customers in this dataset.
- **Basic subscription** customers churn slightly more than Standard or Premium.

See `outputs/categorical_churn_rates.png`, `outputs/numeric_features_by_churn.png`, and `outputs/correlation_heatmap.png` for the full picture.

## License

MIT — feel free to use and adapt.

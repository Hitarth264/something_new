"""
Run predictions on new customer data using the trained churn model.

Usage:
    python src/predict.py --input path/to/new_customers.csv --output predictions.csv

Input CSV must contain the same columns as the training data (minus 'Churn'):
    Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
    Subscription Type, Contract Length, Total Spend, Last Interaction
    (CustomerID is optional and will be carried through to the output if present)
"""
import argparse
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument("--input", required=True, help="Path to input CSV with customer data.")
    parser.add_argument("--output", default="predictions.csv", help="Path to write predictions CSV.")
    parser.add_argument("--model", default="models/churn_model.joblib", help="Path to trained model.")
    args = parser.parse_args()

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.input)

    customer_ids = df["CustomerID"] if "CustomerID" in df.columns else None
    features = df.drop(columns=["CustomerID"], errors="ignore")
    features = features.drop(columns=["Churn"], errors="ignore")  # in case labels are present

    preds = pipe.predict(features)
    probas = pipe.predict_proba(features)[:, 1]

    out = pd.DataFrame({
        "Churn_Prediction": preds,
        "Churn_Probability": probas.round(4),
    })
    if customer_ids is not None:
        out.insert(0, "CustomerID", customer_ids)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()

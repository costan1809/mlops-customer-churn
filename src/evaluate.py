import pandas as pd
import yaml
import os
import sys
import json
import joblib
import mlflow

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    params_path = sys.argv[1]
    params = load_params(params_path)

    test_data_path = params["preprocess"]["output_test"]
    model_path = params["train"]["model_output"]
    metrics_output = params["evaluate"]["metrics_output"]

    # Load test data
    df = pd.read_csv(test_data_path)

    X_test = df.drop(columns=["churn"])
    y_test = df["churn"]

    # Load trained model
    model = joblib.load(model_path)

    # Predictions
    preds = model.predict(X_test)

    # Compute metrics
    test_accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    metrics = {
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Save metrics JSON
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)

    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=4)

    # Log to MLflow (attach to latest run)
    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metrics(metrics)

    print(f"Saved metrics -> {metrics_output}")
    print("Predicted class distribution:")
    print(pd.Series(preds).value_counts())


if __name__ == "__main__":
    main()
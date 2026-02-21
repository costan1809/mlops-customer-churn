import pandas as pd
import yaml
import os
import sys
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    params_path = sys.argv[1]
    params = load_params(params_path)

    train_data_path = params["preprocess"]["output_train"]
    model_type = params["train"]["model_type"]
    random_state = params["train"]["random_state"]
    model_output = params["train"]["model_output"]

    # Load training data
    df = pd.read_csv(train_data_path)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Model selection
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"],
            random_state=random_state
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError("Unsupported model_type")

    # --------------------------
    # MLflow setup
    # --------------------------

    mlflow.set_experiment("customer_churn_dvc_mlflow")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)

        if model_type == "random_forest":
            mlflow.log_param("n_estimators", params["train"]["n_estimators"])
            mlflow.log_param("max_depth", params["train"]["max_depth"])

        # Train model
        model.fit(X, y)

        # Training accuracy
        train_preds = model.predict(X)
        train_accuracy = accuracy_score(y, train_preds)

        # Log metric
        mlflow.log_metric("train_accuracy", train_accuracy)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(__file__)

        # Save model locally (for DVC)
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        joblib.dump(model, model_output)

        print(f"Trained model -> {model_output}")


if __name__ == "__main__":
    main()
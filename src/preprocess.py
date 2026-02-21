import pandas as pd
import yaml
import os
import sys
from sklearn.model_selection import train_test_split


def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    params_path = sys.argv[1]
    params = load_params(params_path)

    raw_data_path = params["generate"]["output"]
    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]
    output_train = params["preprocess"]["output_train"]
    output_test = params["preprocess"]["output_test"]

    # Load raw data
    df = pd.read_csv(raw_data_path)

    # Drop customer_id (not a feature)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Recombine features + target
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_train), exist_ok=True)

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print(f"Saved train -> {output_train}, test -> {output_test}")


if __name__ == "__main__":
    main()
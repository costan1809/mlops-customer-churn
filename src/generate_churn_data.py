import pandas as pd
import numpy as np
import yaml
import os
import sys


def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def generate_data(n_samples, churn_rate):
    np.random.seed(42)

    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n_samples + 1)]

    tenure_months = np.random.randint(0, 73, size=n_samples)

    contract_type = np.random.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_samples,
        p=[0.6, 0.25, 0.15]
    )

    is_senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    partner = np.random.choice([0, 1], size=n_samples)
    dependents = np.random.choice([0, 1], size=n_samples)

    has_phone_service = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])

    has_internet_service = np.random.choice(
        ["none", "dsl", "fiber"],
        size=n_samples,
        p=[0.1, 0.4, 0.5]
    )

    has_online_security = np.random.choice([0, 1], size=n_samples)
    has_tech_support = np.random.choice([0, 1], size=n_samples)
    streaming_tv = np.random.choice([0, 1], size=n_samples)
    streaming_movies = np.random.choice([0, 1], size=n_samples)

    monthly_charges = np.random.normal(70, 30, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)

    total_charges = tenure_months * monthly_charges

    payment_method = np.random.choice(
        ["credit-card", "bank-transfer", "electronic-check", "mailed-check"],
        size=n_samples
    )

    paperless_billing = np.random.choice([0, 1], size=n_samples)

    # -----------------------------
    # Churn probability logic
    # -----------------------------

    churn_prob = np.full(n_samples, churn_rate)

    churn_prob += np.where(contract_type == "month-to-month", 0.15, 0)
    churn_prob -= np.where(contract_type == "two-year", 0.15, 0)

    churn_prob += np.where(has_internet_service == "fiber", 0.1, 0)

    churn_prob -= tenure_months / 200

    churn_prob += (monthly_charges - 70) / 300

    churn_prob -= np.where(has_tech_support == 1, 0.1, 0)
    churn_prob -= np.where(has_online_security == 1, 0.05, 0)

    churn_prob = np.clip(churn_prob, 0.05, 0.95)

    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "is_senior_citizen": is_senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "has_phone_service": has_phone_service,
        "has_internet_service": has_internet_service,
        "has_online_security": has_online_security,
        "has_tech_support": has_tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "churn": churn
    })

    return df


def main():
    params_path = sys.argv[1]
    params = load_params(params_path)

    n_samples = params["generate"]["n_samples"]
    churn_rate = params["generate"]["churn_rate"]
    output_path = params["generate"]["output"]

    df = generate_data(n_samples, churn_rate)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated churn data -> {output_path}")


if __name__ == "__main__":
    main()
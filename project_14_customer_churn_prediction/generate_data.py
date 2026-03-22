"""
Generate a realistic telecom customer churn dataset with 5,000 customers.
Churn correlates with month-to-month contracts, high charges, short tenure,
electronic check payment, and lack of support services.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

customer_id = [f"CID-{str(i).zfill(5)}" for i in range(1, N + 1)]
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])

# Contract type drives many downstream features
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21]
)

# Tenure correlates with contract type
tenure_months = np.zeros(N, dtype=int)
for i in range(N):
    if contract[i] == "Month-to-month":
        tenure_months[i] = int(np.clip(np.random.exponential(18), 1, 72))
    elif contract[i] == "One year":
        tenure_months[i] = int(np.clip(np.random.normal(36, 15), 1, 72))
    else:
        tenure_months[i] = int(np.clip(np.random.normal(48, 14), 1, 72))

# Phone and internet services
phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No",
    "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58]),
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]
)

# Add-on services (only if internet service exists)
def internet_addon(internet_service, yes_prob=0.35):
    result = []
    for inet in internet_service:
        if inet == "No":
            result.append("No internet service")
        else:
            result.append(np.random.choice(["Yes", "No"], p=[yes_prob, 1 - yes_prob]))
    return np.array(result)

online_security = internet_addon(internet_service, 0.30)
online_backup = internet_addon(internet_service, 0.33)
device_protection = internet_addon(internet_service, 0.33)
tech_support = internet_addon(internet_service, 0.30)
streaming_tv = internet_addon(internet_service, 0.38)
streaming_movies = internet_addon(internet_service, 0.38)

# Billing
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.60, 0.40])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N,
    p=[0.34, 0.23, 0.22, 0.21],
)

# Monthly charges based on services
monthly_charges = np.zeros(N)
for i in range(N):
    base = 20.0
    if phone_service[i] == "Yes":
        base += np.random.uniform(0, 5)
    if multiple_lines[i] == "Yes":
        base += np.random.uniform(5, 10)
    if internet_service[i] == "DSL":
        base += np.random.uniform(10, 25)
    elif internet_service[i] == "Fiber optic":
        base += np.random.uniform(25, 50)
    for svc in [online_security[i], online_backup[i], device_protection[i],
                tech_support[i], streaming_tv[i], streaming_movies[i]]:
        if svc == "Yes":
            base += np.random.uniform(5, 12)
    monthly_charges[i] = round(base + np.random.normal(0, 3), 2)

monthly_charges = np.clip(monthly_charges, 18.0, 120.0).round(2)

# Total charges = monthly * tenure with some noise
total_charges = (monthly_charges * tenure_months * np.random.uniform(0.95, 1.05, N)).round(2)

# --- Churn logic ---
# Base churn probability, then adjust by risk factors
churn_prob = np.full(N, 0.05)

# Month-to-month is the biggest driver
churn_prob = np.where(contract == "Month-to-month", churn_prob + 0.30, churn_prob)
churn_prob = np.where(contract == "One year", churn_prob + 0.05, churn_prob)
churn_prob = np.where(contract == "Two year", churn_prob - 0.03, churn_prob)

# Short tenure increases churn strongly
churn_prob = np.where(tenure_months <= 6, churn_prob + 0.20, churn_prob)
churn_prob = np.where((tenure_months > 6) & (tenure_months <= 12), churn_prob + 0.10, churn_prob)
churn_prob = np.where(tenure_months >= 48, churn_prob - 0.10, churn_prob)

# Fiber optic (higher charges, more issues)
churn_prob = np.where(internet_service == "Fiber optic", churn_prob + 0.12, churn_prob)
churn_prob = np.where(internet_service == "No", churn_prob - 0.08, churn_prob)

# Electronic check payment
churn_prob = np.where(payment_method == "Electronic check", churn_prob + 0.12, churn_prob)
churn_prob = np.where(payment_method == "Credit card (automatic)", churn_prob - 0.05, churn_prob)

# No support services increases churn
churn_prob = np.where(online_security == "No", churn_prob + 0.06, churn_prob)
churn_prob = np.where(tech_support == "No", churn_prob + 0.06, churn_prob)

# High monthly charges
churn_prob = np.where(monthly_charges > 80, churn_prob + 0.07, churn_prob)
churn_prob = np.where(monthly_charges > 100, churn_prob + 0.05, churn_prob)

# Senior citizens churn slightly more
churn_prob = np.where(senior_citizen == 1, churn_prob + 0.06, churn_prob)

# Paperless billing
churn_prob = np.where(paperless_billing == "Yes", churn_prob + 0.04, churn_prob)

# No partner/dependents
churn_prob = np.where((partner == "No") & (dependents == "No"), churn_prob + 0.05, churn_prob)

# Clip and generate churn
churn_prob = np.clip(churn_prob, 0.02, 0.85)

# Calibrate to hit ~26% overall churn rate
# Adjust the scale so the mean is around 0.26
current_mean = churn_prob.mean()
churn_prob = churn_prob * (0.26 / current_mean)
churn_prob = np.clip(churn_prob, 0.02, 0.90)

churn = np.array(["Yes" if np.random.random() < p else "No" for p in churn_prob])

print(f"Generated churn rate: {(churn == 'Yes').mean():.3f}")

# Introduce ~30 missing values in total_charges (realistic)
missing_idx = np.random.choice(N, 30, replace=False)
total_charges_series = pd.Series(total_charges)
total_charges_series.iloc[missing_idx] = np.nan

df = pd.DataFrame({
    "customer_id": customer_id,
    "gender": gender,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure_months,
    "contract": contract,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges_series,
    "phone_service": phone_service,
    "multiple_lines": multiple_lines,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "churn": churn,
})

df.to_csv("data/telco_churn.csv", index=False)
print(f"Saved {len(df)} rows to data/telco_churn.csv")
print(f"Columns: {list(df.columns)}")
print(f"Missing total_charges: {df['total_charges'].isna().sum()}")

# Verify correlations
mtm_churn = (df[df["contract"] == "Month-to-month"]["churn"] == "Yes").mean()
ty_churn = (df[df["contract"] == "Two year"]["churn"] == "Yes").mean()
print(f"Month-to-month churn rate: {mtm_churn:.3f}")
print(f"Two year churn rate: {ty_churn:.3f}")
print(f"Ratio: {mtm_churn / ty_churn:.1f}x")

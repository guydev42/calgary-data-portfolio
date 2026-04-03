"""
Generate a realistic SaaS usage analytics dataset with 3,000 users
and 6 months of daily activity logs.
Churn correlates with declining logins, low feature adoption,
high support tickets, free tier, and low NPS.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 3000

# --- User profiles ---
user_id = [f"USR-{str(i).zfill(5)}" for i in range(1, N + 1)]

# Company assignment (~600 companies, some with multiple users)
n_companies = 600
company_pool = [f"CMP-{str(i).zfill(4)}" for i in range(1, n_companies + 1)]
company_id = np.random.choice(company_pool, N)

plan_tier = np.random.choice(
    ["Free", "Pro", "Enterprise"], N, p=[0.40, 0.38, 0.22]
)

industries = [
    "Technology", "Healthcare", "Finance", "Education", "Marketing",
    "E-commerce", "Manufacturing", "Consulting", "Media", "Logistics",
]
industry = np.random.choice(industries, N)

# Signup dates spread over the past 12-24 months
base_date = datetime(2025, 10, 1)
signup_offsets = np.random.randint(180, 730, N)
signup_date = [base_date - timedelta(days=int(d)) for d in signup_offsets]

# --- Usage metrics (monthly aggregates representing the most recent month) ---

# Daily logins depend on plan tier
daily_logins = np.zeros(N)
for i in range(N):
    if plan_tier[i] == "Enterprise":
        daily_logins[i] = np.clip(np.random.normal(18, 5), 1, 40)
    elif plan_tier[i] == "Pro":
        daily_logins[i] = np.clip(np.random.normal(12, 5), 0, 35)
    else:
        daily_logins[i] = np.clip(np.random.exponential(5), 0, 25)
daily_logins = daily_logins.round(1)

# Features used (out of 25 available features)
features_used = np.zeros(N, dtype=int)
for i in range(N):
    if plan_tier[i] == "Enterprise":
        features_used[i] = int(np.clip(np.random.normal(18, 4), 3, 25))
    elif plan_tier[i] == "Pro":
        features_used[i] = int(np.clip(np.random.normal(12, 4), 2, 25))
    else:
        features_used[i] = int(np.clip(np.random.exponential(4), 1, 15))

# Session duration in minutes
session_duration_min = np.zeros(N)
for i in range(N):
    if plan_tier[i] == "Enterprise":
        session_duration_min[i] = np.clip(np.random.normal(35, 10), 5, 90)
    elif plan_tier[i] == "Pro":
        session_duration_min[i] = np.clip(np.random.normal(22, 8), 3, 70)
    else:
        session_duration_min[i] = np.clip(np.random.exponential(10), 1, 45)
session_duration_min = session_duration_min.round(1)

# Actions per session
actions_per_session = np.zeros(N)
for i in range(N):
    base_actions = 3 + features_used[i] * 0.8 + np.random.normal(0, 3)
    actions_per_session[i] = np.clip(base_actions, 1, 50)
actions_per_session = actions_per_session.round(1)

# Last active date and days since last login
observation_date = datetime(2025, 10, 1)
days_since_last_login = np.zeros(N, dtype=int)
for i in range(N):
    if plan_tier[i] == "Enterprise":
        days_since_last_login[i] = int(np.clip(np.random.exponential(3), 0, 60))
    elif plan_tier[i] == "Pro":
        days_since_last_login[i] = int(np.clip(np.random.exponential(7), 0, 90))
    else:
        days_since_last_login[i] = int(np.clip(np.random.exponential(14), 0, 120))

last_active_date = [
    observation_date - timedelta(days=int(d)) for d in days_since_last_login
]

# Monthly active days (out of 30)
monthly_active_days = np.zeros(N, dtype=int)
for i in range(N):
    if daily_logins[i] > 15:
        monthly_active_days[i] = int(np.clip(np.random.normal(25, 3), 5, 30))
    elif daily_logins[i] > 8:
        monthly_active_days[i] = int(np.clip(np.random.normal(18, 5), 3, 30))
    elif daily_logins[i] > 3:
        monthly_active_days[i] = int(np.clip(np.random.normal(10, 4), 1, 25))
    else:
        monthly_active_days[i] = int(np.clip(np.random.exponential(4), 0, 15))

# Support tickets (last 6 months)
support_tickets = np.zeros(N, dtype=int)
for i in range(N):
    if plan_tier[i] == "Free":
        support_tickets[i] = int(np.clip(np.random.poisson(2.5), 0, 15))
    elif plan_tier[i] == "Pro":
        support_tickets[i] = int(np.clip(np.random.poisson(1.5), 0, 12))
    else:
        support_tickets[i] = int(np.clip(np.random.poisson(1.0), 0, 10))

# NPS score (0-10)
nps_score = np.zeros(N, dtype=int)
for i in range(N):
    if plan_tier[i] == "Enterprise":
        nps_score[i] = int(np.clip(np.random.normal(8.0, 1.5), 0, 10))
    elif plan_tier[i] == "Pro":
        nps_score[i] = int(np.clip(np.random.normal(7.0, 2.0), 0, 10))
    else:
        nps_score[i] = int(np.clip(np.random.normal(5.5, 2.5), 0, 10))

# --- Churn logic ---
churn_prob = np.full(N, 0.05)

# Free tier is the biggest churn driver
churn_prob = np.where(plan_tier == "Free", churn_prob + 0.25, churn_prob)
churn_prob = np.where(plan_tier == "Pro", churn_prob + 0.05, churn_prob)
churn_prob = np.where(plan_tier == "Enterprise", churn_prob - 0.03, churn_prob)

# Declining logins (low daily logins)
churn_prob = np.where(daily_logins <= 2, churn_prob + 0.25, churn_prob)
churn_prob = np.where((daily_logins > 2) & (daily_logins <= 5), churn_prob + 0.12, churn_prob)
churn_prob = np.where(daily_logins >= 15, churn_prob - 0.10, churn_prob)

# Low feature adoption
churn_prob = np.where(features_used <= 3, churn_prob + 0.20, churn_prob)
churn_prob = np.where((features_used > 3) & (features_used <= 6), churn_prob + 0.08, churn_prob)
churn_prob = np.where(features_used >= 15, churn_prob - 0.08, churn_prob)

# High support tickets
churn_prob = np.where(support_tickets >= 5, churn_prob + 0.15, churn_prob)
churn_prob = np.where(support_tickets >= 3, churn_prob + 0.06, churn_prob)
churn_prob = np.where(support_tickets == 0, churn_prob - 0.04, churn_prob)

# Low NPS
churn_prob = np.where(nps_score <= 3, churn_prob + 0.18, churn_prob)
churn_prob = np.where((nps_score > 3) & (nps_score <= 5), churn_prob + 0.08, churn_prob)
churn_prob = np.where(nps_score >= 9, churn_prob - 0.08, churn_prob)

# Days since last login
churn_prob = np.where(days_since_last_login >= 30, churn_prob + 0.20, churn_prob)
churn_prob = np.where(days_since_last_login >= 14, churn_prob + 0.10, churn_prob)
churn_prob = np.where(days_since_last_login <= 2, churn_prob - 0.05, churn_prob)

# Low session duration
churn_prob = np.where(session_duration_min <= 5, churn_prob + 0.10, churn_prob)

# Low monthly active days
churn_prob = np.where(monthly_active_days <= 3, churn_prob + 0.15, churn_prob)
churn_prob = np.where(monthly_active_days >= 20, churn_prob - 0.06, churn_prob)

# Clip and calibrate to ~22% churn rate
churn_prob = np.clip(churn_prob, 0.02, 0.90)
current_mean = churn_prob.mean()
churn_prob = churn_prob * (0.22 / current_mean)
churn_prob = np.clip(churn_prob, 0.02, 0.92)

is_churned = np.array([1 if np.random.random() < p else 0 for p in churn_prob])

print(f"Generated churn rate: {is_churned.mean():.3f}")

# --- Assemble dataframe ---
df = pd.DataFrame({
    "user_id": user_id,
    "company_id": company_id,
    "plan_tier": plan_tier,
    "signup_date": [d.strftime("%Y-%m-%d") for d in signup_date],
    "industry": industry,
    "daily_logins": daily_logins,
    "features_used": features_used,
    "session_duration_min": session_duration_min,
    "actions_per_session": actions_per_session,
    "last_active_date": [d.strftime("%Y-%m-%d") for d in last_active_date],
    "days_since_last_login": days_since_last_login,
    "monthly_active_days": monthly_active_days,
    "support_tickets": support_tickets,
    "nps_score": nps_score,
    "is_churned": is_churned,
})

df.to_csv("data/saas_usage.csv", index=False)
print(f"Saved {len(df)} rows to data/saas_usage.csv")
print(f"Columns: {list(df.columns)}")

# Verify correlations
free_churn = df[df["plan_tier"] == "Free"]["is_churned"].mean()
ent_churn = df[df["plan_tier"] == "Enterprise"]["is_churned"].mean()
print(f"\nFree tier churn rate: {free_churn:.3f}")
print(f"Enterprise churn rate: {ent_churn:.3f}")
print(f"Ratio: {free_churn / max(ent_churn, 0.001):.1f}x")

low_login = df[df["daily_logins"] <= 2]["is_churned"].mean()
high_login = df[df["daily_logins"] >= 15]["is_churned"].mean()
print(f"Low-login churn rate: {low_login:.3f}")
print(f"High-login churn rate: {high_login:.3f}")

"""
Generate a realistic employee engagement dataset with 8,000 employees across
5 departments and 12 months of recognition data.
Disengagement correlates with low recognition frequency, short tenure,
no peer recognition, high absenteeism, and low survey scores.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000

employee_id = [f"EMP-{str(i).zfill(5)}" for i in range(1, N + 1)]

# Departments with realistic distribution
department = np.random.choice(
    ["Engineering", "Sales", "Marketing", "Operations", "Customer Support"],
    N,
    p=[0.28, 0.24, 0.16, 0.18, 0.14],
)

# Role level influences recognition behavior
role_level = np.random.choice(
    ["Junior", "Mid", "Senior", "Manager"],
    N,
    p=[0.30, 0.35, 0.22, 0.13],
)

# Tenure correlates with role level
tenure_months = np.zeros(N, dtype=int)
for i in range(N):
    if role_level[i] == "Junior":
        tenure_months[i] = int(np.clip(np.random.exponential(12), 1, 72))
    elif role_level[i] == "Mid":
        tenure_months[i] = int(np.clip(np.random.normal(30, 14), 3, 96))
    elif role_level[i] == "Senior":
        tenure_months[i] = int(np.clip(np.random.normal(48, 16), 12, 120))
    else:  # Manager
        tenure_months[i] = int(np.clip(np.random.normal(60, 18), 18, 144))

# --- Recognition events received (12-month totals) ---
# Base recognition depends on role level and department
recognition_received = np.zeros(N)
for i in range(N):
    base = 3.0
    if role_level[i] == "Mid":
        base += 2.0
    elif role_level[i] == "Senior":
        base += 4.0
    elif role_level[i] == "Manager":
        base += 5.0
    if department[i] == "Sales":
        base += 2.0  # Sales has more recognition programs
    elif department[i] == "Customer Support":
        base -= 1.0  # Support often overlooked
    # Tenure bonus
    base += min(tenure_months[i] / 24, 3.0)
    recognition_received[i] = max(0, int(np.random.poisson(base)))

recognition_received = recognition_received.astype(int)

# --- Recognition events given ---
# Managers give more, seniors give moderately
recognition_given = np.zeros(N, dtype=int)
for i in range(N):
    base = 1.5
    if role_level[i] == "Manager":
        base += 6.0
    elif role_level[i] == "Senior":
        base += 3.0
    elif role_level[i] == "Mid":
        base += 1.0
    recognition_given[i] = max(0, int(np.random.poisson(base)))

# --- Reward type ---
reward_type = np.random.choice(
    ["Monetary", "Points", "Badge", "PTO"],
    N,
    p=[0.18, 0.40, 0.30, 0.12],
)

# --- Monthly recognition frequency (avg over 12 months) ---
monthly_recognition_frequency = (recognition_received / 12.0).round(2)

# --- Average reward value ---
avg_reward_value = np.zeros(N)
for i in range(N):
    if reward_type[i] == "Monetary":
        avg_reward_value[i] = round(np.random.uniform(25, 200), 2)
    elif reward_type[i] == "Points":
        avg_reward_value[i] = round(np.random.uniform(5, 50), 2)
    elif reward_type[i] == "Badge":
        avg_reward_value[i] = round(np.random.uniform(0, 10), 2)
    else:  # PTO
        avg_reward_value[i] = round(np.random.uniform(50, 300), 2)

# --- Peer vs manager recognition ratio ---
# High ratio = mostly peer recognition, low = mostly manager recognition
peer_vs_manager_ratio = np.zeros(N)
for i in range(N):
    if recognition_received[i] == 0:
        peer_vs_manager_ratio[i] = 0.0
    else:
        base_ratio = np.random.beta(3, 2)  # skewed toward peers
        if department[i] == "Sales":
            base_ratio *= 1.1  # Sales has more peer shoutouts
        peer_vs_manager_ratio[i] = round(np.clip(base_ratio, 0, 1), 2)

# --- Satisfaction survey score (1-5) ---
# Correlates with recognition, tenure, department
satisfaction_survey = np.zeros(N)
for i in range(N):
    base = 3.2
    base += min(recognition_received[i] / 10, 0.8)
    base += min(tenure_months[i] / 60, 0.5)
    if department[i] == "Engineering":
        base += 0.2
    elif department[i] == "Customer Support":
        base -= 0.3
    satisfaction_survey[i] = round(np.clip(base + np.random.normal(0, 0.6), 1, 5), 1)

# --- Absenteeism days (12-month total) ---
absenteeism_days = np.zeros(N, dtype=int)
for i in range(N):
    base = 4.0
    if satisfaction_survey[i] < 2.5:
        base += 6.0
    elif satisfaction_survey[i] < 3.0:
        base += 3.0
    if tenure_months[i] < 12:
        base += 2.0
    if recognition_received[i] == 0:
        base += 3.0
    absenteeism_days[i] = max(0, int(np.random.poisson(base)))

# --- Engagement score (1-10) ---
# This is the core metric; disengagement defined as score < 4
engagement_score = np.zeros(N)
for i in range(N):
    base = 6.0

    # Recognition frequency is the strongest driver
    if monthly_recognition_frequency[i] >= 1.0:
        base += 1.5
    elif monthly_recognition_frequency[i] >= 0.5:
        base += 0.5
    elif monthly_recognition_frequency[i] == 0:
        base -= 2.0

    # Peer recognition matters
    if recognition_received[i] > 0 and peer_vs_manager_ratio[i] > 0.6:
        base += 0.5
    elif recognition_received[i] > 0 and peer_vs_manager_ratio[i] < 0.2:
        base -= 0.3

    # Tenure effect
    if tenure_months[i] < 6:
        base -= 1.0
    elif tenure_months[i] < 12:
        base -= 0.5
    elif tenure_months[i] > 48:
        base += 0.5

    # Satisfaction alignment
    base += (satisfaction_survey[i] - 3.0) * 0.5

    # Absenteeism drags engagement down
    if absenteeism_days[i] > 12:
        base -= 1.5
    elif absenteeism_days[i] > 8:
        base -= 0.8

    # Reward type effect
    if reward_type[i] == "PTO":
        base += 0.3
    elif reward_type[i] == "Badge":
        base -= 0.2

    # Department culture
    if department[i] == "Engineering":
        base += 0.2
    elif department[i] == "Customer Support":
        base -= 0.4

    # Role level
    if role_level[i] == "Manager":
        base += 0.4
    elif role_level[i] == "Junior":
        base -= 0.3

    engagement_score[i] = round(np.clip(base + np.random.normal(0, 0.8), 1, 10), 1)

# --- Binary target: is_disengaged ---
is_disengaged = (engagement_score < 4).astype(int)

print(f"Generated disengagement rate: {is_disengaged.mean():.3f}")

# Introduce ~50 missing values in satisfaction_survey (realistic survey non-response)
missing_idx = np.random.choice(N, 50, replace=False)
satisfaction_series = pd.Series(satisfaction_survey)
satisfaction_series.iloc[missing_idx] = np.nan

df = pd.DataFrame({
    "employee_id": employee_id,
    "department": department,
    "role_level": role_level,
    "tenure_months": tenure_months,
    "recognition_events_received": recognition_received,
    "recognition_events_given": recognition_given,
    "reward_type": reward_type,
    "monthly_recognition_frequency": monthly_recognition_frequency,
    "avg_reward_value": avg_reward_value,
    "peer_vs_manager_ratio": peer_vs_manager_ratio,
    "engagement_score": engagement_score,
    "satisfaction_survey": satisfaction_series,
    "absenteeism_days": absenteeism_days,
    "is_disengaged": is_disengaged,
})

df.to_csv("data/employee_engagement.csv", index=False)
print(f"Saved {len(df)} rows to data/employee_engagement.csv")
print(f"Columns: {list(df.columns)}")
print(f"Missing satisfaction_survey: {df['satisfaction_survey'].isna().sum()}")

# Verify correlations
low_recog_disengage = df[df["monthly_recognition_frequency"] == 0]["is_disengaged"].mean()
high_recog_disengage = df[df["monthly_recognition_frequency"] >= 1.0]["is_disengaged"].mean()
print(f"\nNo recognition disengagement rate: {low_recog_disengage:.3f}")
print(f"High recognition (>=1/mo) disengagement rate: {high_recog_disengage:.3f}")
print(f"Ratio: {low_recog_disengage / max(high_recog_disengage, 0.001):.1f}x")

short_tenure_disengage = df[df["tenure_months"] < 12]["is_disengaged"].mean()
long_tenure_disengage = df[df["tenure_months"] >= 48]["is_disengaged"].mean()
print(f"\nShort tenure (<12mo) disengagement: {short_tenure_disengage:.3f}")
print(f"Long tenure (>=48mo) disengagement: {long_tenure_disengage:.3f}")

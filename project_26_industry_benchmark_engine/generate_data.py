"""
Generate a realistic industry benchmark dataset with 500 companies across 8 industries.
Each industry has distinct distributions reflecting real-world patterns:
- Tech: higher eNPS, higher revenue per employee, better engagement
- Healthcare: higher turnover, more training hours
- Finance: higher profit margins, larger reward values
- Manufacturing: lower engagement, higher promotion rates
- Retail: lower budgets, higher turnover
- Education: high training hours, lower profit margins
- Energy: high revenue per employee, moderate turnover
- Government: high diversity index, lower eNPS, stable turnover
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500

INDUSTRIES = [
    "Tech", "Finance", "Healthcare", "Manufacturing",
    "Retail", "Education", "Energy", "Government",
]
INDUSTRY_WEIGHTS = [0.18, 0.14, 0.14, 0.12, 0.14, 0.10, 0.10, 0.08]

SIZES = ["Small", "Medium", "Large", "Enterprise"]
SIZE_WEIGHTS = [0.25, 0.35, 0.25, 0.15]

REGIONS = [
    "North America", "Europe", "Asia-Pacific",
    "Latin America", "Middle East & Africa",
]
REGION_WEIGHTS = [0.35, 0.28, 0.22, 0.09, 0.06]

# --- Industry-specific distribution parameters ---
# Format: (mean, std) for each metric per industry
PARAMS = {
    #                     recog_freq  reward_val  budget/emp  turnover  engagement  eNPS   train_hrs  promo_rate  diversity  rev/emp     profit_margin
    "Tech":              ((2.8, 0.9), (120, 40),  (850, 250), (0.14, 0.05), (7.8, 0.8), (42, 18), (32, 10), (0.10, 0.03), (0.62, 0.12), (280000, 80000), (0.18, 0.08)),
    "Finance":           ((2.2, 0.7), (150, 50),  (920, 300), (0.16, 0.06), (7.2, 0.9), (32, 20), (28, 9),  (0.08, 0.03), (0.48, 0.14), (320000, 90000), (0.22, 0.09)),
    "Healthcare":        ((1.8, 0.6), (80, 30),   (520, 180), (0.22, 0.07), (6.8, 1.0), (18, 22), (40, 12), (0.07, 0.03), (0.58, 0.13), (180000, 60000), (0.10, 0.06)),
    "Manufacturing":     ((1.4, 0.5), (70, 25),   (420, 150), (0.18, 0.06), (6.2, 1.1), (15, 20), (24, 8),  (0.09, 0.03), (0.44, 0.12), (200000, 70000), (0.12, 0.07)),
    "Retail":            ((1.6, 0.6), (55, 20),   (320, 120), (0.28, 0.09), (6.0, 1.2), (10, 22), (18, 7),  (0.06, 0.02), (0.52, 0.14), (120000, 45000), (0.06, 0.04)),
    "Education":         ((1.5, 0.5), (65, 22),   (380, 130), (0.15, 0.05), (7.0, 0.9), (25, 18), (45, 14), (0.05, 0.02), (0.60, 0.13), (95000, 35000),  (0.05, 0.04)),
    "Energy":            ((1.9, 0.6), (110, 35),  (750, 220), (0.16, 0.05), (6.8, 0.9), (22, 19), (30, 10), (0.07, 0.02), (0.40, 0.12), (350000, 100000),(0.16, 0.08)),
    "Government":        ((1.2, 0.4), (60, 20),   (400, 140), (0.12, 0.04), (6.4, 1.0), (12, 18), (35, 11), (0.06, 0.02), (0.68, 0.10), (110000, 40000), (0.04, 0.03)),
}

# Employee count ranges by company size
SIZE_EMPLOYEES = {
    "Small":      (20, 150),
    "Medium":     (151, 1000),
    "Large":      (1001, 5000),
    "Enterprise": (5001, 50000),
}

# --- Generate data ---
company_id = [f"CMP-{str(i).zfill(4)}" for i in range(1, N + 1)]
industry = np.random.choice(INDUSTRIES, N, p=INDUSTRY_WEIGHTS)
company_size = np.random.choice(SIZES, N, p=SIZE_WEIGHTS)
region = np.random.choice(REGIONS, N, p=REGION_WEIGHTS)

# Employee count based on size
employee_count = np.zeros(N, dtype=int)
for i in range(N):
    lo, hi = SIZE_EMPLOYEES[company_size[i]]
    employee_count[i] = int(np.random.uniform(lo, hi))

# Generate metrics per industry
avg_recognition_frequency = np.zeros(N)
avg_reward_value = np.zeros(N)
budget_per_employee = np.zeros(N)
turnover_rate = np.zeros(N)
engagement_score = np.zeros(N)
eNPS = np.zeros(N)
training_hours_per_employee = np.zeros(N)
promotion_rate = np.zeros(N)
diversity_index = np.zeros(N)
revenue_per_employee = np.zeros(N)
profit_margin = np.zeros(N)

for i in range(N):
    p = PARAMS[industry[i]]
    avg_recognition_frequency[i] = np.random.normal(*p[0])
    avg_reward_value[i] = np.random.normal(*p[1])
    budget_per_employee[i] = np.random.normal(*p[2])
    turnover_rate[i] = np.random.normal(*p[3])
    engagement_score[i] = np.random.normal(*p[4])
    eNPS[i] = np.random.normal(*p[5])
    training_hours_per_employee[i] = np.random.normal(*p[6])
    promotion_rate[i] = np.random.normal(*p[7])
    diversity_index[i] = np.random.normal(*p[8])
    revenue_per_employee[i] = np.random.normal(*p[9])
    profit_margin[i] = np.random.normal(*p[10])

    # Size adjustments: larger companies have slightly better metrics
    if company_size[i] == "Large":
        budget_per_employee[i] *= 1.08
        avg_recognition_frequency[i] *= 1.05
    elif company_size[i] == "Enterprise":
        budget_per_employee[i] *= 1.15
        avg_recognition_frequency[i] *= 1.10
        revenue_per_employee[i] *= 1.05

# Clip to realistic ranges
avg_recognition_frequency = np.clip(avg_recognition_frequency, 0.2, 6.0).round(2)
avg_reward_value = np.clip(avg_reward_value, 10, 400).round(2)
budget_per_employee = np.clip(budget_per_employee, 50, 2000).round(2)
turnover_rate = np.clip(turnover_rate, 0.02, 0.55).round(3)
engagement_score = np.clip(engagement_score, 3.0, 10.0).round(1)
eNPS = np.clip(eNPS, -50, 90).astype(int)
training_hours_per_employee = np.clip(training_hours_per_employee, 4, 80).round(1)
promotion_rate = np.clip(promotion_rate, 0.01, 0.20).round(3)
diversity_index = np.clip(diversity_index, 0.10, 0.95).round(2)
revenue_per_employee = np.clip(revenue_per_employee, 30000, 600000).astype(int)
profit_margin = np.clip(profit_margin, -0.05, 0.45).round(3)

df = pd.DataFrame({
    "company_id": company_id,
    "industry": industry,
    "company_size": company_size,
    "region": region,
    "employee_count": employee_count,
    "avg_recognition_frequency": avg_recognition_frequency,
    "avg_reward_value": avg_reward_value,
    "budget_per_employee": budget_per_employee,
    "turnover_rate": turnover_rate,
    "engagement_score": engagement_score,
    "eNPS": eNPS,
    "training_hours_per_employee": training_hours_per_employee,
    "promotion_rate": promotion_rate,
    "diversity_index": diversity_index,
    "revenue_per_employee": revenue_per_employee,
    "profit_margin": profit_margin,
})

df.to_csv("data/industry_benchmark.csv", index=False)
print(f"Saved {len(df)} rows to data/industry_benchmark.csv")
print(f"Columns: {list(df.columns)}")
print(f"\nIndustry distribution:")
print(df["industry"].value_counts().sort_index())
print(f"\nSize distribution:")
print(df["company_size"].value_counts())
print(f"\nSample metrics by industry:")
summary = df.groupby("industry").agg({
    "turnover_rate": "mean",
    "engagement_score": "mean",
    "eNPS": "mean",
    "revenue_per_employee": "mean",
    "profit_margin": "mean",
}).round(3)
print(summary)

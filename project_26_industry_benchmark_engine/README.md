<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Industry%20Benchmark%20Engine&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Compare%20recognition%20program%20KPIs%20across%20sectors%20with%20percentile-based%20benchmarking&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Industries-8-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Companies-500-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/KPIs-12-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Percentile%20Engine-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A benchmarking engine that computes industry-standard metrics across sectors, enabling companies to compare their recognition program KPIs against sector peers with percentile-based rankings and gap analysis.**

HR tech platforms need to show clients how their recognition programs compare to industry averages. This project builds a comprehensive benchmarking engine that ingests KPI data from 500 companies across 8 industries, computes percentile rankings within peer groups, performs gap analysis against configurable targets, and surfaces actionable improvement recommendations. The interactive dashboard lets users drill into any company, filter by industry/size/region, and export custom benchmark reports.

```
Problem   →  Companies lack visibility into how their programs compare to peers
Solution  →  Percentile engine with peer filtering, gap analysis, and custom reports
Impact    →  12 KPIs benchmarked across 8 industries with configurable peer groups
```

---

## Key results

| Metric | Value |
|--------|-------|
| Companies benchmarked | 500 |
| Industries covered | 8 |
| KPIs tracked | 12 |
| Peer comparison dimensions | 3 (industry, size, region) |
| Dashboard pages | 5 |

**Key findings**

- **Tech companies** lead in engagement score (7.8/10) and eNPS (42), while **Retail** lags behind (6.0, 10)
- **Healthcare** has the highest turnover rate (22%) but also the most training hours per employee (40 hrs)
- **Finance** leads in profit margin (22%) and reward value ($150), reflecting higher compensation benchmarks
- **Enterprise companies** spend 15% more per employee on recognition than small companies
- Strong positive correlation between recognition frequency and engagement scores across all industries

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Raw data    │───▶│  Data loader      │───▶│  Benchmark engine   │
│  (500 rows)  │    │  + peer filtering │    │  (percentiles,      │
└─────────────┘    └──────────────────┘    │   gap analysis)     │
                                            └──────────┬──────────┘
                                                       │
                          ┌────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Industry summary    │───▶│  Custom report       │
              │  + cross-ranking     │    │  generator           │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Streamlit app       │
                                          │  (5-page dashboard)  │
                                          └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_26_industry_benchmark_engine/
├── data/                  # Benchmark dataset (500 companies)
├── src/                   # Data loading, benchmark engine
├── notebooks/             # EDA, benchmark analysis, segmentation
├── outputs/               # Plots and exported reports
├── app.py                 # Streamlit dashboard (5 pages)
├── generate_data.py       # Synthetic data generator
├── requirements.txt       # Python dependencies
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_26_industry_benchmark_engine

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Run benchmark analysis
python -c "
from src.data_loader import load_data
from src.benchmark import industry_summary, gap_analysis
df = load_data('data/industry_benchmark.csv')
print(industry_summary(df))
print(gap_analysis(df, 'CMP-0001'))
"

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic industry benchmark data |
| Records | 500 companies |
| Industries | 8 (Tech, Finance, Healthcare, Manufacturing, Retail, Education, Energy, Government) |
| KPIs | 12 (recognition, engagement, financial, HR) |
| Dimensions | Industry, company size, region |
| Size categories | Small, Medium, Large, Enterprise |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
</p>

---

## Methodology

<details>
<summary><b>Data generation</b></summary>

- Industry-specific distributions with realistic means and standard deviations
- Company size adjustments (Enterprise companies have higher budgets)
- Regional distribution weighted by market size
</details>

<details>
<summary><b>Percentile engine</b></summary>

- Rank-based percentile computation within configurable peer groups
- Peer filtering by industry, company size, and region
- Cross-industry ranking by any KPI
</details>

<details>
<summary><b>Gap analysis</b></summary>

- Configurable target percentile (P50 to P95)
- Directional gap computation (higher is better vs lower is better)
- Priority classification: High, Medium, On track
</details>

<details>
<summary><b>Dashboard</b></summary>

- Industry Overview: cross-sector heatmap and KPI comparison
- Company Benchmarker: peer comparison with radar chart
- Percentile Rankings: company-level rankings with distributions
- Trend Analysis: scatter plots with OLS trendlines and correlation matrix
- Custom Report Generator: gap analysis with downloadable CSV export
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/). Designed for HR tech platforms benchmarking recognition programs across industries.

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>

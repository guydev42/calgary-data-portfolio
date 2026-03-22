<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=A%2FB%20Test%20Analysis%20Framework&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Rigorous%20frequentist%2C%20Bayesian%2C%20and%20sequential%20experiment%20analysis&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Experiments-3-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/P(B>A)-0.996-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
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

> **A reusable A/B test framework that combines frequentist, Bayesian, and sequential methods to deliver defensible experiment decisions.**

Data-driven organizations run dozens of A/B tests simultaneously, but without rigorous statistical methodology the results are prone to false positives, peeking bias, and multiple comparison errors. This project provides a reusable framework that analyzes controlled experiments using three complementary approaches: frequentist hypothesis testing, Bayesian posterior inference, and sequential monitoring with early stopping. It gives decision-makers clear, defensible recommendations on which changes to ship, with built-in corrections for multiple comparisons.

```
Problem   →  A/B tests are vulnerable to false positives, peeking, and multiple comparison errors
Solution  →  Unified framework with frequentist + Bayesian + sequential analysis
Impact    →  3 experiments analyzed with p<0.001, P(B>A)=0.996, and valid early stopping
```

---

## Key results

| Experiment | p-value | Significant? | Bayesian P(B>A) |
|-----------|---------|-------------|-----------------|
| Website redesign | 0.008 | Yes | 0.996 |
| Pricing change | 0.260 | No | 0.869 |
| Email subject | <0.001 | Yes | 1.000 |

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Experiment      │───▶│  Power           │───▶│  Frequentist     │
│  design          │    │  analysis        │    │  tests           │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Bayesian            │───▶│  Sequential          │
              │  inference           │    │  monitoring          │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Multiple comparison │───▶│  Decision            │
              │  correction          │    │  report              │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_16_ab_test_framework/
├── data/
│   ├── ab_test_results.csv          # Experiment results
│   └── generate_data.py             # Synthetic experiment generator
├── src/
│   ├── __init__.py
│   ├── experiment.py                # Statistical analysis engine
│   └── visualizations.py            # Plot generation
├── notebooks/
│   └── 01_analysis.ipynb            # Full experiment walkthrough
├── figures/
│   ├── conversion_exp_001.png       # Conversion rate comparisons
│   ├── posterior_exp_001.png        # Bayesian posterior distributions
│   ├── sequential_exp_001.png       # Sequential monitoring boundaries
│   ├── sample_size_mde.png          # Power analysis curves
│   └── forest_plot.png              # Cross-experiment summary
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_16_ab_test_framework

# Install dependencies
pip install -r requirements.txt

# Generate experiment data
python data/generate_data.py

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic controlled experiment data |
| Experiments | 3 (website redesign, pricing change, email subject) |
| Methods | Frequentist, Bayesian, sequential |
| Metrics | Conversion rate (proportions), continuous outcomes |
| Correction | Bonferroni, Holm, Benjamini-Hochberg FDR |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Frequentist testing</b></summary>

- Z-test for proportions and Welch's t-test for continuous metrics
- Effect sizes: Cohen's h (proportions) and Cohen's d (means)
- Confidence intervals for all estimates
</details>

<details>
<summary><b>Bayesian A/B testing</b></summary>

- Beta-Binomial model with uninformative priors
- Posterior sampling to compute P(B > A)
- Credible intervals for relative lift
</details>

<details>
<summary><b>Sequential monitoring</b></summary>

- O'Brien-Fleming spending function for valid early stopping
- Controls false positive rate even with continuous monitoring
- Boundary plots for each interim analysis
</details>

<details>
<summary><b>Multiple comparison correction</b></summary>

- Bonferroni correction (family-wise error rate)
- Holm step-down procedure
- Benjamini-Hochberg FDR control
- Applied across all simultaneous experiments
</details>

<details>
<summary><b>Power analysis</b></summary>

- Sample size calculator for future experiment planning
- Inputs: baseline rate, minimum detectable effect, alpha, power
- Sensitivity curves for MDE vs sample size trade-offs
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>

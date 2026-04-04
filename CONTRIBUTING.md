# Contributing

Thanks for your interest in this portfolio. Contributions, suggestions, and bug reports are welcome.

## Getting started

```bash
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio
pip install -r requirements.txt
```

## Project structure

Each of the 26 projects follows this layout:

```
project_XX_name/
├── data/              # Raw and processed datasets
├── docs/              # Architecture documentation
├── models/            # Saved model artifacts (.joblib)
├── notebooks/         # Executed Jupyter notebooks
│   └── clean/         # Output-stripped notebooks for reproducibility
├── src/               # Modular source code
├── tests/             # Unit tests (pytest)
├── app.py             # Streamlit dashboard
├── config.yaml        # Hyperparameters and paths
├── Dockerfile         # Container definition
├── environment.yml    # Conda environment
├── Makefile           # Build automation
├── README.md          # Project documentation with results
└── requirements.txt   # Python dependencies
```

## Running tests

```bash
# All projects
python -m pytest project_*/tests/ -v

# Single project
cd project_01_building_permit_cost_predictor
python -m pytest tests/ -v
```

## Running a project dashboard

```bash
cd project_01_building_permit_cost_predictor
pip install -r requirements.txt
streamlit run app.py
```

## Guidelines

- Run tests before submitting a PR
- Keep notebook outputs committed (clean versions live in `notebooks/clean/`)
- Update the project README if metrics change
- Large data files (>10 MB) should not be committed — use `.gitignore`

## Reporting issues

Open an issue on GitHub with:
- Which project is affected
- Steps to reproduce
- Expected vs actual behavior

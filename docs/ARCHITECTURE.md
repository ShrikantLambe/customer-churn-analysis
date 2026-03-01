# Project Architecture: Customer Churn Analysis

## Overview
This project is structured for clarity, modularity, and production-readiness. It supports reproducible research, robust model development, and easy deployment.

## Directory Structure
```
customer-churn-analysis/
│
├── app/                # Streamlit app for interactive prediction
├── data/               # Raw and processed datasets (not versioned)
├── docs/               # Project documentation (architecture, design, etc.)
├── models/             # Saved model files (not versioned)
├── notebooks/          # Jupyter notebooks for EDA and prototyping
├── src/                # Core source code (modular, testable)
├── tests/              # Unit and integration tests
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore rules
├── config.yaml         # Centralized configuration
├── CONTRIBUTING.md     # Contribution guidelines
├── README.md           # Project overview and instructions
├── requirements.txt    # Python dependencies
```

## src/ Module Layout
- `preprocessing.py` — Data cleaning, feature engineering, and transformation pipelines
- `model_training.py` — Model training, cross-validation, and saving
- `evaluation_utils.py` — Metrics, plots, and explainability tools
- `business_impact.py` — Business value simulation and reporting
- `logging_utils.py` — Logging setup for reproducibility
- `config_utils.py` — Config loading and random seed control
- `shap_explain.py` — SHAP explainability for model interpretation

## Best Practices
- All data and model artifacts are excluded from git via `.gitignore`.
- Notebooks are used for EDA and prototyping, scripts for automation and reproducibility.
- All configuration is centralized in `config.yaml`.
- Tests are in `tests/` and should be run before every commit.
- Documentation is in `docs/` and `README.md`.

## CI/CD & Deployment (Recommended)
- Add GitHub Actions for linting, testing, and build checks.
- Use DVC or MLflow for model versioning in production.
- Deploy the Streamlit app via Streamlit Cloud or Docker.

---

For more details, see the README.md and CONTRIBUTING.md.

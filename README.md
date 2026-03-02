# Customer Churn Analysis

A production-ready, modular, and well-documented machine learning project for predicting customer churn using the Telco Customer Churn dataset.

## 🚀 Project Features
- End-to-end ML workflow: EDA, preprocessing, training, evaluation, business impact simulation, and deployment
- Modular codebase with clear separation of concerns
- Streamlit app for interactive churn prediction
- Reproducible experiments with config and logging
- Unit tests and best practices for production

## 🚀 Production-Ready Features

### Versioned Models
- Models, preprocessors, and feature names are saved with version tags (e.g., `v1`, `v2`).
- Update `model_version` in `config.yaml` to control versioning.

### CI Automation
- GitHub Actions workflow in `.github/workflows/ci.yml` runs linting and tests on every push/PR.

### Business Workflow Integration
- A FastAPI microservice is provided in `api/app.py` for programmatic prediction:
  ```bash
  uvicorn api.app:app --reload
  ```
- POST to `/predict` with a JSON dict of features matching the training columns.

## 📁 Project Structure
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed breakdown.

## 🗂️ Key Directories
- `src/` — Core source code (preprocessing, training, evaluation, utils)
- `notebooks/` — Jupyter notebooks for EDA and prototyping
- `app/` — Streamlit app for interactive prediction
- `models/` — Saved model files (not versioned in git)
- `data/` — Raw and processed datasets (not versioned in git)
- `tests/` — Unit and integration tests
- `docs/` — Architecture and design documentation

## 🛠️ Tech Stack
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, shap, streamlit, joblib

## ⚙️ Setup & Installation
1. Clone the repo and create a virtual environment:
   ```bash
   git clone <repo-url>
   cd customer-churn-analysis
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download the Telco Customer Churn dataset and place it in `data/`.
3. Run tests:
   ```bash
   python -m unittest discover tests
   ```
4. Train and save a model:
   ```bash
   python train_and_save_model.py
   ```
5. Simulate business impact:
   ```bash
   python run_business_impact.py
   ```
6. Launch the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🧪 Testing
- All new code should be covered by unit tests in `tests/`.
- Run tests with `python -m unittest discover tests` or `pytest`.

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License
MIT License

## Installation (Development Mode)

1. Ensure you have Python 3.8+ and pip installed.
2. From the project root, install the package in editable mode:

```bash
pip install -e .
```

This will make the `src` package importable from anywhere in your project, including Streamlit apps.

## Running the Streamlit App

**Important:** Always run Streamlit from the project root directory, _not_ from inside the `app/` folder. This ensures Python can find the `src` package.

From the project root, run:

```bash
streamlit run app/streamlit_app.py
```

If you run Streamlit from inside the `app/` directory, you will get `ModuleNotFoundError: No module named 'src'`.

## Project Structure

```
customer-churn-analysis/
├── src/
│   ├── config_utils.py
│   ├── ...
├── app/
│   └── streamlit_app.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Notes
- No sys.path hacks are needed.
- All imports like `from src.config_utils import load_config` will work after installation and when running from the root.
- For production, you can build and distribute the package using standard Python packaging tools.

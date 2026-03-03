# Customer Churn Analysis

A production-ready, modular, and well-documented machine learning and AI project for predicting customer churn using the Telco Customer Churn dataset.

## 🚀 Project Features
- End-to-end ML workflow: EDA, preprocessing, training, evaluation, business impact simulation, and deployment
- Modular codebase with clear separation of concerns
- Streamlit app for interactive churn prediction and advanced AI features
- LLM-powered Copilot for conversational queries, retention strategy, and explainability
- Reproducible experiments with config and logging
- Unit tests and best practices for production

## 🤖 Advanced AI Features
- **Conversational Data Query:** Ask natural language questions about churn data
- **Retention Strategy Advisor:** AI-powered recommendations for customer retention
- **Executive Churn Report:** Auto-generated business reports with LLM explanations
- **Copilot Panel:** Chat with an AI assistant for business and technical queries
- **Explainability:** SHAP integration and LLM explanations for model predictions
- **Persona Generation:** Segmentation and persona cards for actionable insights
- **Robust Error Handling:** UI and backend handle missing data, columns, and segment queries gracefully

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
- `src/` — Core source code (preprocessing, training, evaluation, utils, LLM modules)
- `notebooks/` — Jupyter notebooks for EDA and prototyping
- `app/` — Streamlit app for interactive prediction and Copilot
- `models/` — Saved model files (not versioned in git)
- `data/` — Raw and processed datasets (not versioned in git)
- `tests/` — Unit and integration tests
- `docs/` — Architecture and design documentation

## 🛠️ Tech Stack
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, shap, streamlit, joblib, openai, pyyaml, fastapi

## ⚙️ Setup & Installation
1. Clone the repo and create a virtual environment:
   ```bash
   git clone <repo-url>
   cd customer-churn-analysis
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download the Telco Customer Churn dataset and place it in `data/` as either `churn_data.csv` or `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Set your OpenAI API key in your environment:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
4. Run tests:
   ```bash
   python -m unittest discover tests
   ```
5. Train and save a model:
   ```bash
   python train_and_save_model.py
   ```
6. Simulate business impact:
   ```bash
   python run_business_impact.py
   ```
7. Launch the Streamlit app:
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

## Troubleshooting & FAQ
- **ModuleNotFoundError:** Always run Streamlit from the project root, not inside `app/`.
- **Dataset not found:** Place your CSV in `data/` as `churn_data.csv` or `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **OpenAI API errors:** Ensure your API key is set and you have internet access.
- **Copilot/LLM not responding:** Check your API key and rate limits.

## Notes
- No sys.path hacks are needed.
- All imports like `from src.config_utils import load_config` will work after installation and when running from the root.
- For production, you can build and distribute the package using standard Python packaging tools.

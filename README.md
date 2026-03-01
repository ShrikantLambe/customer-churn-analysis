# Customer Churn Analysis

A production-ready, modular, and well-documented machine learning project for predicting customer churn using the Telco Customer Churn dataset.

## 🚀 Project Features
- End-to-end ML workflow: EDA, preprocessing, training, evaluation, business impact simulation, and deployment
- Modular codebase with clear separation of concerns
- Streamlit app for interactive churn prediction
- Reproducible experiments with config and logging
- Unit tests and best practices for production

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

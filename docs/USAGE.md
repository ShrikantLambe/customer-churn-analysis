# Usage Guide: Customer Churn Analysis

## 1. Environment Setup
- Clone the repository and create a virtual environment:
  ```bash
  git clone <repo-url>
  cd customer-churn-analysis
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## 2. Data Preparation
- Download the Telco Customer Churn dataset from Kaggle and place it in the `data/` directory.
- Or run the provided script:
  ```bash
  python download_dataset.py
  ```

## 3. Exploratory Data Analysis (EDA)
- Open and run the notebook:
  ```bash
  jupyter notebook notebooks/eda_telco_churn.ipynb
  ```

## 4. Model Training
- Train and save a model:
  ```bash
  python train_and_save_model.py
  ```

## 5. Business Impact Simulation
- Run the business impact analysis:
  ```bash
  python run_business_impact.py
  ```

## 6. Streamlit App
- Launch the app for interactive prediction:
  ```bash
  streamlit run app/streamlit_app.py
  ```

## 7. Testing
- Run all tests:
  ```bash
  python -m unittest discover tests
  ```

## 8. CI/CD
- GitHub Actions will automatically run tests on every push or pull request to `main`.

---

For more details, see [docs/ARCHITECTURE.md](ARCHITECTURE.md) and [README.md](../README.md).

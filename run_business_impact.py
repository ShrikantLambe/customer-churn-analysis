"""
Script to run business impact simulation for churn prediction
"""
import pandas as pd
import pickle
from src.preprocessing import preprocess_data, load_data
from src.model_training import evaluate_model
from src.business_impact import simulate_business_impact

# Load data
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = load_data(DATA_PATH)

# Load config (if available)
try:
    from src.config_utils import load_config
    config = load_config()
except ImportError:
    config = None

# Preprocess data
if config:
    X_train, X_test, y_train, y_test = preprocess_data(df, config)
else:
    X_train, X_test, y_train, y_test = preprocess_data(df)

# Load trained model (update path if needed)
MODEL_PATH = "models/best_churn_model_v1.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Predict churn probabilities on test set
if hasattr(model, 'predict_proba'):
    y_proba = model.predict_proba(X_test)[:, 1]
else:
    raise ValueError("Model does not support probability prediction.")

# Run business impact simulation
simulate_business_impact(y_test, y_proba)

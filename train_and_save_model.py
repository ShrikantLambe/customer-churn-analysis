"""
Script to train and save a baseline churn prediction model
"""
import pandas as pd
import pickle
from src.preprocessing import preprocess_data, load_data
from src.model_training import train_logistic_regression

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

# Train logistic regression model
if config:
    model, cv_score = train_logistic_regression(X_train, y_train, config)
else:
    model, cv_score = train_logistic_regression(X_train, y_train)

print(f"Logistic Regression CV ROC-AUC: {cv_score:.4f}")

# Save model
MODEL_PATH = "models/best_churn_model_v1.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_PATH}")

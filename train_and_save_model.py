"""
Script to train and save a baseline churn prediction model
"""
import yaml
import pandas as pd
import pickle
import os
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

# Save feature names for use in Streamlit app
if config:
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        df, config, return_preprocessor=True, return_feature_names=True)
else:
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        df, return_preprocessor=True, return_feature_names=True)
FEATURE_NAMES_PATH = "models/feature_names_v1.pkl"
with open(FEATURE_NAMES_PATH, "wb") as f:
    pickle.dump(feature_names, f)

# Train logistic regression model
if config:
    model, cv_score = train_logistic_regression(X_train, y_train, config)
else:
    model, cv_score = train_logistic_regression(X_train, y_train)

print(f"Logistic Regression CV ROC-AUC: {cv_score:.4f}")


# --- Versioned model saving ---
config_path = "config.yaml"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    version = config_yaml.get('output', {}).get('model_version', 'v1')
else:
    version = 'v1'
MODEL_PATH = f"models/best_churn_model_{version}.pkl"
PREPROCESSOR_PATH = f"models/preprocessor_{version}.pkl"
FEATURE_NAMES_PATH = f"models/feature_names_{version}.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(PREPROCESSOR_PATH, "wb") as f:
    pickle.dump(preprocessor, f)
with open(FEATURE_NAMES_PATH, "wb") as f:
    pickle.dump(feature_names, f)
print(f"Model saved to {MODEL_PATH}")
print(f"Preprocessor saved to {PREPROCESSOR_PATH}")
print(f"Feature names saved to {FEATURE_NAMES_PATH}")

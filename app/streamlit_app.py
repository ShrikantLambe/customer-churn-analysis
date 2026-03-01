"""
Streamlit app for Customer Churn Prediction
"""
import streamlit as st
import numpy as np
import pickle
import logging
import os
from src.config_utils import load_config

# --- UI Section ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.markdown("---")
st.header("Enter Customer Details")

tenure = st.number_input(
    "Tenure (months)", min_value=0, max_value=100, value=12)
contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"])

# --- Preprocessing for model input ---


def preprocess_input(tenure, contract, monthly_charges, internet_service):
    # Example encoding, adjust as per your model's pipeline
    contract_map = {"Month-to-month": [1, 0, 0],
                    "One year": [0, 1, 0], "Two year": [0, 0, 1]}
    internet_map = {"DSL": [1, 0, 0],
                    "Fiber optic": [0, 1, 0], "No": [0, 0, 1]}
    features = [tenure, monthly_charges] + \
        contract_map[contract] + internet_map[internet_service]
    return np.array(features).reshape(1, -1)

# --- Load Model ---


@st.cache_resource
def load_model():
    config = load_config()
    model_dir = config['output']['model_dir']
    version = config['output']['model_version']
    model_path = os.path.join(model_dir, f"best_churn_model_{version}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Loaded model from {model_path}")
    return model


model = load_model()

# --- Prediction ---
if st.button("Predict Churn Probability"):
    X_input = preprocess_input(
        tenure, contract, monthly_charges, internet_service)
    prob = model.predict_proba(X_input)[0, 1]
    # Risk category
    if prob < 0.2:
        risk = "Low"
        color = "green"
    elif prob < 0.5:
        risk = "Medium"
        color = "orange"
    else:
        risk = "High"
        color = "red"
    st.markdown("---")
    st.header("Prediction Result")
    st.markdown(f"**Churn Probability:** {prob:.2%}")
    st.markdown(
        f"**Risk Category:** <span style='color:{color}; font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
    st.info(
        "Interpretation: High risk customers may benefit from targeted retention offers.")

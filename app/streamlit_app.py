from config_utils import load_config
import streamlit as st
import pickle
import os
import pandas as pd

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.header("Enter Customer Details")

# Group features for better UX
demographic_cols, service_cols, billing_cols = st.columns([1, 1, 1])

with demographic_cols:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12)

with service_cols:
    st.subheader("Services")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox(
        "Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox(
        "Online Security", ["No internet service", "No", "Yes"])
    OnlineBackup = st.selectbox(
        "Online Backup", ["No internet service", "No", "Yes"])
    DeviceProtection = st.selectbox(
        "Device Protection", ["No internet service", "No", "Yes"])
    TechSupport = st.selectbox(
        "Tech Support", ["No internet service", "No", "Yes"])
    StreamingTV = st.selectbox(
        "Streaming TV", ["No internet service", "No", "Yes"])
    StreamingMovies = st.selectbox(
        "Streaming Movies", ["No internet service", "No", "Yes"])

with billing_cols:
    st.subheader("Billing & Contract")
    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    MonthlyCharges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

# --- Load model, preprocessor, and feature names ---


@st.cache_resource
def load_model_preprocessor_and_features():
    config = load_config()
    model_dir = config['output']['model_dir']
    version = config['output']['model_version']
    model_path = os.path.join(model_dir, f"best_churn_model_{version}.pkl")
    preprocessor_path = os.path.join(model_dir, f"preprocessor_{version}.pkl")
    feature_names_path = os.path.join(model_dir, "feature_names_v1.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(feature_names_path, "rb") as f:
        feature_names = pickle.load(f)
    return model, preprocessor, feature_names


model, preprocessor, feature_names = load_model_preprocessor_and_features()

# --- Prediction ---
center_col = st.columns([1, 2, 1])[1]
with center_col:
    if st.button("Predict Churn Probability", use_container_width=True):
        # Build input in the correct order
        input_data = [
            gender,
            SeniorCitizen,
            Partner,
            Dependents,
            tenure,
            PhoneService,
            MultipleLines,
            InternetService,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            Contract,
            PaperlessBilling,
            PaymentMethod,
            MonthlyCharges,
            TotalCharges
        ]
        X_input_df = pd.DataFrame(
            [input_data], columns=feature_names
        )
        X_processed = preprocessor.transform(X_input_df)
        prob = model.predict_proba(X_processed)[0, 1]
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
        st.markdown(
            f"**Churn Probability:** <span style='font-size:1.5em'>{prob:.2%}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Risk Category:** <span style='color:{color}; font-weight:bold; "
            f"font-size:1.3em'>{risk}</span>",
            unsafe_allow_html=True
        )
        st.info(
            "Interpretation: High risk customers may benefit from targeted "
            "retention offers."
        )

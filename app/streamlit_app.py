from chat_memory import ChatMemory
from ai_router import route_intent
from persona_generator import generate_persona_card
from segmentation import run_kmeans
from model_debugger import model_debugger
from retention_strategy import retention_strategy
from executive_report import generate_executive_report
from query_agent import query_agent
from llm_explainer import llm_explain
from feature_parser import parse_features
import pickle
from config_utils import load_config
import streamlit as st
import pandas as pd
import os
# Load churn dataset at the very top
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/churn_data.csv')
ALT_DATA_PATH = os.path.join(os.path.dirname(
    __file__), '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
elif os.path.exists(ALT_DATA_PATH):
    df = pd.read_csv(ALT_DATA_PATH)
else:
    df = pd.DataFrame()


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
        # Retention Strategy Advisor
        st.markdown("---")
        st.subheader("Retention Strategy Advisor")
        # Example: risk drivers (replace with actual SHAP logic)
        risk_drivers = ["tenure", "MonthlyCharges", "Contract"]
        customer_profile = dict(zip(feature_names, input_data))
        strategy = retention_strategy(customer_profile, prob, risk_drivers)
        st.markdown(f"**Urgency:** {strategy.get('risk_level', '')}")
        st.markdown("**Recommended Actions:**")
        for action in strategy.get('recommended_actions', []):
            st.markdown(f"- {action}")
        st.markdown(f"**Rationale:** {strategy.get('rationale', '')}")

st.markdown("---")
st.subheader("What-If Scenario Simulator")
user_query = st.text_input(
    "Type a what-if scenario (e.g. 'What if tenure is 24 months?')",
    ""
)
if user_query:
    # Build current features as dict
    customer_features = dict(zip(feature_names, input_data))
    # Define feature schema for parser
    feature_schema = {name: type(
        val).__name__ for name, val in customer_features.items()}
    parsed = parse_features(user_query, feature_schema)
    if "error" in parsed:
        st.error(f"LLM parser error: {parsed['error']}")
    else:
        # Update features
        new_features = customer_features.copy()
        new_features.update(parsed)
        # Build new input for model
        new_input = [new_features[name] for name in feature_names]
        X_new_df = pd.DataFrame([new_input], columns=feature_names)
        X_new_processed = preprocessor.transform(X_new_df)
        new_prob = model.predict_proba(X_new_processed)[0, 1]
        percent_change = ((new_prob - prob) / prob * 100) if prob != 0 else 0
        st.markdown(f"**Original Probability:** {prob:.2%}")
        st.markdown(f"**New Probability:** {new_prob:.2%}")
        st.markdown(f"**Change:** {percent_change:+.2f}%")
        # Get SHAP values for new input (example, replace with actual SHAP logic)
        # top5_shap = ...
        st.markdown("---")
        st.subheader("LLM Explanation of Change")
        # Use previous top5_shap or dummy if not available
        explanation_md = llm_explain(new_features, new_prob, [])
        st.markdown(explanation_md, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Conversational Data Query Assistant")
data_query = st.text_input(
    "Ask a question about churn data (e.g. 'Which segment has highest churn?')",
    ""
)
if data_query:
    # Assume df is loaded and available
    result, img_b64, explanation, error = query_agent(data_query, df)
    if error:
        st.error(error)
    else:
        if result is not None:
            st.markdown("**Result Table:**")
            st.dataframe(result)
        if img_b64:
            st.markdown("**Auto-generated Plot:**")
            st.image(f"data:image/png;base64,{img_b64}")
        if explanation:
            st.markdown("**LLM Insight:**")
            st.markdown(explanation, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Executive Churn Report")
if st.button("Generate Executive Churn Report", use_container_width=True):
    # Example: gather required metrics (replace with actual project logic)
    churn_rate = df['Churn'].mean() if 'Churn' in df.columns else 0.0
    metrics = {
        'accuracy': 0.85,  # replace with actual
        'recall': 0.78,    # replace with actual
        'precision': 0.80  # replace with actual
    }
    # Example SHAP values (replace with actual global SHAP)
    top5_shap = [("tenure", 0.15), ("MonthlyCharges", 0.12), ("Contract",
                                                              0.10), ("InternetService", 0.08), ("TechSupport", 0.07)]
    high_risk_summary = "Segment: Month-to-month, Fiber optic, High charges. Churn rate: 52%."
    report_md = generate_executive_report(
        churn_rate, metrics, top5_shap, high_risk_summary)
    st.markdown(report_md, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Model Diagnostics AI Assistant")
# Example: replace with actual confusion matrix and metrics
conf_matrix = [[80, 20], [15, 85]]  # [[TN, FP], [FN, TP]]
metrics = {
    'accuracy': 0.85,
    'precision': 0.80,
    'recall': 0.78
}
imbalance_ratio = 0.3  # e.g. minority/majority class ratio
diag_md = model_debugger(conf_matrix, metrics, imbalance_ratio)
st.markdown(diag_md, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Customer Persona Cards from Segmentation")
# Example: select features for clustering
cluster_features = ["tenure", "MonthlyCharges", "Contract"]
if st.button("Generate Customer Personas", use_container_width=True):
    df_clustered, summaries = run_kmeans(df, cluster_features, n_clusters=4)
    for summary in summaries:
        persona_md = generate_persona_card(summary)
        st.markdown(persona_md, unsafe_allow_html=True)

# Layout: main app + right-side chat panel
main_col, chat_col = st.columns([3, 1])

with main_col:
    # ...existing main app code...
    pass

with chat_col:
    st.markdown("## AI Retention Copilot Panel")
    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = ChatMemory()
    chat_memory = st.session_state["chat_memory"]
    user_input = st.text_area("Ask the AI Copilot anything...", "")
    if st.button("Send", key="send_chat"):  # Unique key for chat button
        # Simple intent router (replace with NLP intent detection)
        if "what if" in user_input.lower():
            intent = "what_if"
            # Fill with actual schema
            payload = {"query": user_input, "feature_schema": {}}
        elif "explain" in user_input.lower():
            intent = "explain_prediction"
            payload = {"features": {}, "prob": 0.0,
                       "shap_top5": []}  # Fill with actual
        elif "query" in user_input.lower() or "show" in user_input.lower():
            intent = "query_data"
            payload = {"query": user_input, "df": df}
        elif "report" in user_input.lower():
            intent = "generate_report"
            payload = {"churn_rate": 0.0, "metrics": {}, "top5_shap": [
            ], "high_risk_summary": ""}  # Fill with actual
        elif "recommend" in user_input.lower() or "action" in user_input.lower():
            intent = "recommend_actions"
            payload = {"profile": {}, "prob": 0.0,
                       "risk_drivers": []}  # Fill with actual
        elif "diagnostic" in user_input.lower() or "debug" in user_input.lower():
            intent = "model_diagnostics"
            payload = {"conf_matrix": [], "metrics": {},
                       "imbalance_ratio": 0.0}  # Fill with actual
        elif "persona" in user_input.lower() or "segment" in user_input.lower():
            intent = "generate_personas"
            payload = {"df": df, "features": []}  # Fill with actual
        else:
            intent = "unknown"
            payload = {}
        chat_memory.add_message("user", user_input)
        response = route_intent(intent, payload, chat_memory)
        chat_memory.add_message("assistant", str(response))
    for msg in chat_memory.get_history():
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

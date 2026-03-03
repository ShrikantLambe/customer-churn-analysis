import streamlit as st
import pandas as pd
from utils.config_utils import load_config
from models.model_manager import load_model, load_preprocessor, load_feature_names
from services.business_logic import calculate_risk_category, summarize_high_risk_segment
from llm.llm_explainer import llm_explain
from feature_parser import parse_features
from query_agent import query_agent
from executive_report import generate_executive_report
from retention_strategy import retention_strategy
from model_debugger import model_debugger
from segmentation import run_kmeans
from persona_generator import generate_persona_card
from ai_router import route_intent
from chat_memory import ChatMemory
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

config = load_config()
model_dir = config['output']['model_dir']
version = config['output']['model_version']
model = load_model(model_dir, version)
preprocessor = load_preprocessor(model_dir, version)
feature_names = load_feature_names(model_dir)

# Load churn dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/churn_data.csv')
ALT_DATA_PATH = os.path.join(os.path.dirname(
    __file__), '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
elif os.path.exists(ALT_DATA_PATH):
    df = pd.read_csv(ALT_DATA_PATH)
else:
    st.warning(
        "Churn dataset not found. Please add your data file to data/churn_data.csv or data/WA_Fn-UseC_-Telco-Customer-Churn.csv.")
    df = pd.DataFrame()

# --- Sidebar Filters ---
with st.sidebar:
    st.title("Customer Filters")
    st.markdown("Select customer details below:")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12)
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
    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
                                 "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)
    st.markdown("---")
    st.markdown(
        "**Tip:** Adjust filters to simulate different customer profiles.")

input_data = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
              DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
X_input_df = pd.DataFrame([input_data], columns=feature_names)
X_processed = preprocessor.transform(X_input_df)
prob = model.predict_proba(X_processed)[0, 1]
risk, color = calculate_risk_category(prob)
customer_profile = dict(zip(feature_names, input_data))

# --- Main Panel ---
st.title("📊 Customer Churn AI Copilot")
st.markdown("<style>body{background-color:#f8f9fa;} .stApp{font-family: 'Inter', sans-serif;} .stTitle{color:#2c3e50;} .stHeader{color:#2980b9;} .stSubheader{color:#34495e;} .stMarkdown{font-size:1.1em;} .stInfo{background:#eaf6fb;}</style>", unsafe_allow_html=True)

st.header("Prediction Result")
st.markdown(
    f"**Churn Probability:** <span style='font-size:1.5em'>{prob:.2%}</span>", unsafe_allow_html=True)
st.markdown(
    f"**Risk Category:** <span style='color:{color}; font-weight:bold; font-size:1.3em'>{risk}</span>", unsafe_allow_html=True)
st.info("Interpretation: High risk customers may benefit from targeted retention offers.")

st.markdown("---")
st.subheader("Retention Strategy Advisor")
risk_drivers = ["tenure", "MonthlyCharges", "Contract"]
strategy = retention_strategy(customer_profile, prob, risk_drivers)
st.markdown(f"**Urgency:** {strategy.get('risk_level', '')}")
st.markdown("**Recommended Actions:**")
for action in strategy.get('recommended_actions', []):
    if isinstance(action, dict):
        st.markdown(
            f"- **{action.get('action', '').capitalize()}** _(Urgency: {action.get('urgency', '')})_  ")
        impact = action.get('expected_impact', action.get('description', ''))
        st.markdown(f"  {impact}")
    else:
        st.markdown(f"- {action}")
st.markdown(f"**Rationale:** {strategy.get('rationale', '')}")

st.markdown("---")
# --- What-If and Data Query Side by Side ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("What-If Scenario Simulator")
    user_query = st.text_input(
        "Type a what-if scenario (e.g. 'What if tenure is 24 months?')", "")
    if user_query:
        feature_schema = {name: type(
            val).__name__ for name, val in customer_profile.items()}
        parsed = parse_features(user_query, feature_schema)
        if "error" in parsed:
            st.error(f"LLM parser error: {parsed['error']}")
        else:
            new_features = customer_profile.copy()
            new_features.update(parsed)
            new_input = [new_features[name] for name in feature_names]
            X_new_df = pd.DataFrame([new_input], columns=feature_names)
            X_new_processed = preprocessor.transform(X_new_df)
            new_prob = model.predict_proba(X_new_processed)[0, 1]
            percent_change = ((new_prob - prob) / prob *
                              100) if prob != 0 else 0
            st.markdown(f"**Original Probability:** {prob:.2%}")
            st.markdown(f"**New Probability:** {new_prob:.2%}")
            st.markdown(f"**Change:** {percent_change:+.2f}%")
            explanation_md = llm_explain(new_features, new_prob, [])
            st.markdown(explanation_md, unsafe_allow_html=True)
with col2:
    st.subheader("Conversational Data Query Assistant")
    data_query = st.text_input(
        "Ask a question about churn data (e.g. 'Which segment has highest churn?')", "")
    if data_query:
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
# --- Executive Report, Diagnostics, Personas Side by Side ---
colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Executive Churn Report")
    if st.button("Generate Executive Churn Report", use_container_width=True):
        churn_rate = df['Churn'].mean() if 'Churn' in df.columns else 0.0
        metrics = {'accuracy': 0.85, 'recall': 0.78, 'precision': 0.80}
        top5_shap = [("tenure", 0.15), ("MonthlyCharges", 0.12), ("Contract",
                                                                  0.10), ("InternetService", 0.08), ("TechSupport", 0.07)]
        high_risk_summary = summarize_high_risk_segment(df)
        report_md = generate_executive_report(
            churn_rate, metrics, top5_shap, high_risk_summary)
        st.markdown(report_md, unsafe_allow_html=True)
with colB:
    st.subheader("Model Diagnostics AI Assistant")
    if st.button("Generate Model Diagnostics", use_container_width=True):
        conf_matrix = [[80, 20], [15, 85]]
        metrics = {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.78}
        imbalance_ratio = 0.3
        diag_md = model_debugger(conf_matrix, metrics, imbalance_ratio)
        st.markdown(diag_md, unsafe_allow_html=True)
with colC:
    st.subheader("Customer Persona Cards from Segmentation")
    cluster_features = ["tenure", "MonthlyCharges", "Contract"]
    if st.button("Generate Customer Personas", use_container_width=True):
        try:
            df_clustered, summaries = run_kmeans(
                df, cluster_features, n_clusters=4)
            for summary in summaries:
                persona_md = generate_persona_card(summary)
                st.markdown(persona_md, unsafe_allow_html=True)
        except KeyError as e:
            st.error(f"Customer segmentation failed: {e}")

# --- AI Copilot Chat Panel ---
st.markdown("---")
st.subheader("AI Retention Copilot Panel")
if "chat_memory" not in st.session_state:
    st.session_state["chat_memory"] = ChatMemory()
chat_memory = st.session_state["chat_memory"]
user_input = st.text_area("Ask the AI Copilot anything...", "")
if st.button("Send", key="send_chat"):
    # Simple intent router (replace with NLP intent detection)
    user_lower = user_input.lower()
    if "what if" in user_lower:
        intent = "what_if"
        payload = {"query": user_input, "feature_schema": {}}
    elif "explain" in user_lower:
        intent = "explain_prediction"
        payload = {"features": {}, "prob": 0.0, "shap_top5": []}
    elif "query" in user_lower or "show" in user_lower:
        intent = "query_data"
        payload = {"query": user_input, "df": df}
    elif "report" in user_lower:
        intent = "generate_report"
        payload = {"churn_rate": 0.0, "metrics": {},
                   "top5_shap": [], "high_risk_summary": ""}
    elif ("recommend" in user_lower or "action" in user_lower or "retain" in user_lower or "retention" in user_lower):
        intent = "recommend_actions"
        # Build a profile from sidebar selections for retention questions
        profile = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }
        # Dummy values for probability and risk drivers
        prob = 0.5
        risk_drivers = ["Contract", "MonthlyCharges"]
        payload = {"profile": profile, "prob": prob,
                   "risk_drivers": risk_drivers}
    elif "diagnostic" in user_lower or "debug" in user_lower:
        intent = "model_diagnostics"
        payload = {"conf_matrix": [], "metrics": {}, "imbalance_ratio": 0.0}
    elif "persona" in user_lower or "segment" in user_lower:
        intent = "generate_personas"
        payload = {"df": df, "features": []}
    else:
        intent = "unknown"
        payload = {}
    chat_memory.add_message("user", user_input)
    response = route_intent(intent, payload, chat_memory)
    chat_memory.add_message("assistant", str(response))
for msg in chat_memory.get_history():
    if msg["role"] == "assistant":
        import json
        resp = msg["content"]
        # Try to parse both single and double quote JSON
        try:
            if resp.startswith("{") and "recommended_actions" in resp:
                # Replace single quotes with double quotes, but avoid breaking inner quotes
                import re
                # Replace only outer single quotes
                resp_json = re.sub(
                    r"(?<=\{|, )'([^']+)'(?=:|,|\})", r'"\1"', resp)
                # Also replace key names
                resp_json = re.sub(
                    r"'([a-zA-Z0-9_]+)'(?=:)", r'"\1"', resp_json)
                data = json.loads(resp_json)
                st.markdown(f"**Retention Strategy**")
                st.markdown(f"**Risk Level:** {data.get('risk_level', 'N/A')}")
                st.markdown("**Recommended Actions:**")
                for act in data.get('recommended_actions', []):
                    st.markdown(
                        f"- **{act.get('action', '')}** _(Urgency: {act.get('urgency', '')})_  ")
                    impact = act.get('expected_impact',
                                     act.get('description', ''))
                    st.markdown(f"  {impact}")
                st.markdown(f"**Rationale:** {data.get('rationale', '')}")
            else:
                st.markdown(f"**Assistant:** {resp}")
        except Exception:
            st.markdown(f"**Assistant:** {resp}")
    else:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

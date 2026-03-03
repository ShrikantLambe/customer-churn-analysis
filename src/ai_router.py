from llm_explainer import llm_explain
from feature_parser import parse_features
from query_agent import query_agent
from executive_report import generate_executive_report
from retention_strategy import retention_strategy
from model_debugger import model_debugger
from persona_generator import generate_persona_card
from segmentation import run_kmeans


def route_intent(intent, payload, session):
    if intent == "explain_prediction":
        return llm_explain(payload["features"], payload["prob"], payload["shap_top5"])
    elif intent == "what_if":
        parsed = parse_features(payload["query"], payload["feature_schema"])
        return parsed
    elif intent == "query_data":
        return query_agent(payload["query"], payload["df"])
    elif intent == "generate_report":
        return generate_executive_report(
            payload["churn_rate"], payload["metrics"], payload["top5_shap"], payload["high_risk_summary"])
    elif intent == "recommend_actions":
        return retention_strategy(
            payload["profile"], payload["prob"], payload["risk_drivers"])
    elif intent == "model_diagnostics":
        return model_debugger(
            payload["conf_matrix"], payload["metrics"], payload["imbalance_ratio"])
    elif intent == "generate_personas":
        df_clustered, summaries = run_kmeans(
            payload["df"], payload["features"], payload.get("n_clusters", 4))
        return [generate_persona_card(s) for s in summaries]
    else:
        return "Intent not recognized."

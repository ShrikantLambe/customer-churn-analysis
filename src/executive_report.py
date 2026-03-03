import os
import openai


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_report_prompt(churn_rate, metrics, top5_shap, high_risk_summary):
    metrics_str = (
        f"Accuracy: {metrics['accuracy']:.2%}\n"
        f"Recall: {metrics['recall']:.2%}\n"
        f"Precision: {metrics['precision']:.2%}"
    )
    shap_str = "\n".join(
        [f"- {feat}: {impact:.3f}" for feat, impact in top5_shap])
    user_prompt = (
        f"Overall churn rate: {churn_rate:.2%}\n"
        f"Model metrics:\n{metrics_str}\n"
        f"Top 5 SHAP features:\n{shap_str}\n"
        f"High-risk segment summary: {high_risk_summary}\n\n"
        "Generate a 1-page executive summary for a VP. Include a revenue risk estimate and 3 strategic recommendations. "
        "Use concise, strategic, non-technical language. Format as markdown."
    )
    system_prompt = (
        "You are a senior business consultant. Given churn analytics, write a concise, strategic executive summary for a VP. "
        "Include revenue risk and recommendations."
    )
    return system_prompt, user_prompt


def generate_executive_report(churn_rate, metrics, top5_shap, high_risk_summary):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_report_prompt(
            churn_rate, metrics, top5_shap, high_risk_summary)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0.5
        )
        markdown = response.choices[0].message.content.strip()
        return markdown
    except Exception as e:
        return (
            "### Executive Report Unavailable\n"
            "Sorry, the report could not be generated. "
            f"Error: {str(e)}"
        )

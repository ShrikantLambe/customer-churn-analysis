import os
import openai


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_prompt(customer_features, churn_prob, shap_top5):
    feature_str = "\n".join(
        [f"- {k}: {v}" for k, v in customer_features.items()])
    shap_str = "\n".join(
        [f"- {feat}: {impact:.3f}" for feat, impact in shap_top5])
    user_prompt = (
        f"Customer features:\n{feature_str}\n\n"
        f"Churn probability: {churn_prob:.2%}\n"
        f"Top 5 SHAP contributions:\n{shap_str}\n\n"
        "Explain the churn risk in business-friendly language. "
        "List the top 3 churn drivers and 2 actionable retention recommendations."
    )
    system_prompt = (
        "You are an expert business analyst. "
        "Given customer features, churn probability, and SHAP values, "
        "write a clear, concise markdown explanation for business users. "
        "Highlight churn drivers and retention actions."
    )
    return system_prompt, user_prompt


def llm_explain(customer_features, churn_prob, shap_top5):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_prompt(
            customer_features, churn_prob, shap_top5)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        markdown = response.choices[0].message.content.strip()
        return markdown
    except Exception as e:
        return (
            "### Prediction Explanation Unavailable\n"
            "Sorry, the LLM explainer could not generate a response. "
            f"Error: {str(e)}"
        )

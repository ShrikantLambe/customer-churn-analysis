import os
import openai


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_persona_prompt(cluster_summary):
    summary_str = (
        f"Cluster {cluster_summary['cluster']}\n"
        f"Size: {cluster_summary['size']}\n"
        f"Feature means: {cluster_summary['means']}\n"
        f"Churn rate: {cluster_summary['churn_rate']:.2%}"
    )
    user_prompt = (
        f"{summary_str}\n\nGenerate a customer persona card with:\n- Persona name\n- Behavioral description\n- Churn risk tendency\n- Retention strategy\nFormat as markdown."
    )
    system_prompt = (
        "You are a customer segmentation expert. Given cluster stats, generate a persona card for business users."
    )
    return system_prompt, user_prompt


def generate_persona_card(cluster_summary):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_persona_prompt(cluster_summary)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.6
        )
        markdown = response.choices[0].message.content.strip()
        return markdown
    except Exception as e:
        return (
            f"### Persona Card Unavailable\nError: {str(e)}"
        )

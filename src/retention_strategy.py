import os
import openai


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_strategy_prompt(profile, churn_prob, risk_drivers):
    profile_str = "\n".join([f"- {k}: {v}" for k, v in profile.items()])
    risk_str = ", ".join(risk_drivers)
    user_prompt = (
        f"Customer profile:\n{profile_str}\n"
        f"Churn probability: {churn_prob:.2%}\n"
        f"Risk drivers: {risk_str}\n\n"
        "Recommend retention actions: discount, contract restructuring, support escalation, upsell. "
        "Classify urgency (Low/Medium/High). Provide expected impact narrative. "
        "Return a JSON object with risk_level, recommended_actions, rationale."
    )
    system_prompt = (
        "You are a retention strategy expert. Given customer profile, churn probability, and risk drivers, "
        "recommend actionable strategies, classify urgency, and explain expected impact. Return structured JSON."
    )
    return system_prompt, user_prompt


def retention_strategy(profile, churn_prob, risk_drivers):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_strategy_prompt(
            profile, churn_prob, risk_drivers)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.5
        )
        import json
        import re
        match = re.search(
            r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return {
                "risk_level": "",
                "recommended_actions": [],
                "rationale": "No valid JSON returned by LLM."
            }
    except Exception as e:
        return {
            "risk_level": "",
            "recommended_actions": [],
            "rationale": f"Error: {str(e)}"
        }

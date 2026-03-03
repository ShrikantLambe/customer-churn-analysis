import os
import openai


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_debug_prompt(conf_matrix, metrics, imbalance_ratio):
    cm_str = f"Confusion matrix: {conf_matrix}"
    metrics_str = (
        f"Accuracy: {metrics['accuracy']:.2%}\n"
        f"Precision: {metrics['precision']:.2%}\n"
        f"Recall: {metrics['recall']:.2%}"
    )
    user_prompt = (
        f"{cm_str}\n"
        f"Model metrics:\n{metrics_str}\n"
        f"Class imbalance ratio: {imbalance_ratio:.2f}\n\n"
        "Analyze model weaknesses. Suggest threshold tuning, resampling, alternative algorithms, and feature engineering improvements. "
        "Format explanation for ML engineers."
    )
    system_prompt = (
        "You are an ML model debugging expert. Given confusion matrix, metrics, and imbalance ratio, "
        "analyze weaknesses and suggest improvements for ML engineers."
    )
    return system_prompt, user_prompt


def model_debugger(conf_matrix, metrics, imbalance_ratio):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_debug_prompt(
            conf_matrix, metrics, imbalance_ratio)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.4
        )
        markdown = response.choices[0].message.content.strip()
        return markdown
    except Exception as e:
        return (
            "### Model Diagnostics Unavailable\n"
            "Sorry, the diagnostics could not be generated. "
            f"Error: {str(e)}"
        )

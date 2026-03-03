import os
import openai
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

SAFE_GLOBALS = {"pd": pd}
SAFE_LOCALS = {}

# Only allow 'df' and pandas/matplotlib
ALLOWED_NAMES = {"df", "pd", "plt"}


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_query_prompt(user_query):
    system_prompt = (
        "You are a Python data assistant. Given a user question and a pandas DataFrame called df, "
        "write safe pandas code to answer the question. Only use df, pd, and matplotlib. "
        "Do not use imports, file I/O, or system calls. Return only the code as a markdown code block."
    )
    user_prompt = f"User question: {user_query}\nDataFrame: df (churn data)"
    return system_prompt, user_prompt


def extract_code_from_response(response_text):
    import re
    match = re.search(r'```python\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'```(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1)
    return response_text.strip()


def safe_execute(code, df):
    # Restrict builtins and globals
    safe_globals = {k: v for k, v in SAFE_GLOBALS.items()
                    if k in ALLOWED_NAMES}
    safe_globals["df"] = df
    safe_globals["plt"] = plt
    safe_locals = {}
    # Only allow assignment to result, fig
    exec_vars = {"result": None, "fig": None}
    try:
        exec(code, safe_globals, exec_vars)
        result = exec_vars.get("result")
        fig = exec_vars.get("fig")
        return result, fig, None
    except Exception as e:
        # Custom error for missing 'segment' or empty DataFrame
        if "segment" in str(e):
            # Suggest valid segments based on key columns
            segment_cols = ["Contract", "InternetService",
                            "gender", "SeniorCitizen"]
            available = [col for col in segment_cols if col in df.columns]
            suggestions = []
            for col in available:
                unique_vals = df[col].unique()
                suggestions.append(
                    f"{col}: {', '.join(str(v) for v in unique_vals)}")
            msg = "The required segment could not be found.\n" \
                  "Try asking about one of these segments:\n" + \
                "\n".join(suggestions)
            return None, None, msg
        if "KeyError" in str(e):
            return None, None, "A required column is missing from the dataset. Please check your data."
        if "EmptyDataError" in str(e) or "No data" in str(e):
            return None, None, "The dataset appears to be empty. Please upload valid data."
        return None, None, str(e)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_b64


def query_agent(user_query, df):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_query_prompt(user_query)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        code = extract_code_from_response(response.choices[0].message.content)
        result, fig, error = safe_execute(code, df)
        if error:
            return None, None, None, f"Code execution error: {error}"
        img_b64 = fig_to_base64(fig) if fig else None
        # Get LLM explanation
        explain_prompt = (
            "Explain the insight from the result and plot in business-friendly language."
        )
        explain_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": explain_prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        explanation = explain_response.choices[0].message.content.strip()
        return result, img_b64, explanation, None
    except Exception as e:
        return None, None, None, str(e)

import os
import openai
import re


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment variables.")
    return api_key


def build_parser_prompt(user_query, feature_schema):
    schema_str = "\n".join(
        [f"- {name}: {ftype}" for name, ftype in feature_schema.items()])
    system_prompt = (
        "You are a data assistant. Parse the user's what-if scenario and return a JSON object "
        "with only the modified features and their new values. Use the provided feature schema. "
        "If a value is ambiguous, ask for clarification."
    )
    user_prompt = (
        f"Feature schema:\n{schema_str}\n\n"
        f"User query: {user_query}\n"
        "Return only the changed features as JSON."
    )
    return system_prompt, user_prompt


def parse_features(user_query, feature_schema):
    try:
        openai.api_key = get_openai_api_key()
        system_prompt, user_prompt = build_parser_prompt(
            user_query, feature_schema)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        # Extract JSON from response
        import json
        match = re.search(
            r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            # Validate keys
            for k in parsed:
                if k not in feature_schema:
                    raise ValueError(f"Unknown feature: {k}")
            return parsed
        else:
            raise ValueError("No JSON found in LLM response.")
    except Exception as e:
        return {"error": str(e)}

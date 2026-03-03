def calculate_risk_category(prob):
    if prob < 0.2:
        return "Low", "green"
    elif prob < 0.5:
        return "Medium", "orange"
    else:
        return "High", "red"


def summarize_high_risk_segment(df):
    # Check for required columns
    if 'Contract' not in df.columns or 'InternetService' not in df.columns:
        return "Required columns ('Contract', 'InternetService') not found in dataset."
    segment = df[(df['Contract'] == 'Month-to-month') &
                 (df['InternetService'] == 'Fiber optic')]
    churn_rate = segment['Churn'].mean() if 'Churn' in segment.columns else 0.0
    return f"Segment: Month-to-month, Fiber optic, High charges. Churn rate: {churn_rate:.2%}."

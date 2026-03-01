"""
SHAP explainability utilities for Random Forest churn model
"""
import shap
import matplotlib.pyplot as plt


def explain_with_shap_rf(model, X, feature_names=None, max_display=15):
    """
    Generate SHAP summary plot for a fitted Random Forest model.
    Args:
        model: Trained RandomForestClassifier
        X: Data used for SHAP value calculation (numpy array or DataFrame)
        feature_names: List of feature names (optional, recommended for interpretability)
        max_display: Number of top features to display
    """
    # Create SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values[1] corresponds to the positive class (churn)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[1], X, feature_names=feature_names, max_display=max_display, show=False
    )
    plt.title('SHAP Summary Plot: Top Features Driving Churn')
    plt.tight_layout()
    plt.show()

    # Comments for interpretation:
    print("""
    SHAP summary plot interpretation:
    - Each point represents a customer (row) and its SHAP value for a feature.
    - Features are ranked by importance (top = most impactful for churn prediction).
    - Color shows feature value (red = high, blue = low).
    - SHAP value (x-axis) shows impact on model output: right = higher churn risk, left = lower.
    - Example: If 'tenure' is blue on the right, low tenure increases churn risk.
    Use this plot to identify which features most influence churn and how their values affect predictions.
    """)

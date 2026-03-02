"""
Business impact simulation for churn prediction model
"""
import numpy as np


def simulate_business_impact(y_true, y_proba, monthly_revenue=70, intervention_effect=0.3, top_pct=0.2):
    """
    Simulate business impact of targeting high-risk customers for retention.
    Args:
        y_true (array-like): True churn labels (0/1)
        y_proba (array-like): Predicted churn probabilities
        monthly_revenue (float): Average monthly revenue per customer
        intervention_effect (float): Fractional reduction in churn due to intervention (e.g., 0.3 = 30%)
        top_pct (float): Proportion of customers to target (e.g., 0.2 = top 20%)
    """
    n_customers = len(y_true)
    n_target = int(n_customers * top_pct)
    # Get indices of top N% high-risk customers
    top_idx = np.argsort(y_proba)[-n_target:][::-1]
    # How many actual churners are in the targeted group?
    actual_churners_in_top = np.sum(y_true[top_idx])
    # Revenue saved: number of churners retained * monthly revenue * intervention effect
    monthly_revenue_saved = (
        actual_churners_in_top * monthly_revenue * intervention_effect
    )
    annual_revenue_saved = monthly_revenue_saved * 12

    print("--- Business Impact Simulation ---")
    print(f"Total customers: {n_customers}")
    print(f"Top {int(top_pct*100)}% high-risk customers targeted: {n_target}")
    print(f"Actual churners in targeted group: {actual_churners_in_top}")
    print(f"If intervention reduces churn by {int(intervention_effect*100)}%:")
    print(
        f"  → Estimated monthly revenue saved: ${monthly_revenue_saved:,.2f}"
    )
    print(
        f"  → Estimated annual revenue saved:  ${annual_revenue_saved:,.2f}"
    )
    print("\nBusiness Explanation:")
    print(
        f"By focusing retention efforts on the top {int(top_pct*100)}% of "
        f"customers most likely to churn, the business can proactively reach "
        f"{actual_churners_in_top} at-risk customers each month. "
        f"If a retention campaign reduces churn by {int(intervention_effect*100)}% "
        f"in this group, the company could save approximately "
        f"${monthly_revenue_saved:,.0f} per month, or "
        f"${annual_revenue_saved:,.0f} per year in recurring revenue. "
        f"This quantifies the financial value of predictive churn modeling "
        f"and targeted interventions."
    )

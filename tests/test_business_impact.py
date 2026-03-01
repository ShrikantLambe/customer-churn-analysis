import unittest
import numpy as np
from src.business_impact import simulate_business_impact


class TestBusinessImpact(unittest.TestCase):
    def test_simulate_business_impact(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_proba = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
        # Should not raise any exceptions
        simulate_business_impact(
            y_true, y_proba, monthly_revenue=50, intervention_effect=0.5, top_pct=0.4)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.evaluation_utils import get_feature_importance


class TestEvaluationUtils(unittest.TestCase):
    def test_get_feature_importance(self):
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        model = RandomForestClassifier().fit(X, y)
        # Should not raise and should print top features
        get_feature_importance(model, feature_names=['A', 'B', 'C'], top_n=2)


if __name__ == '__main__':
    unittest.main()

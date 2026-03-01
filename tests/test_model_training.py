import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.model_training import evaluate_model


class TestModelTraining(unittest.TestCase):
    def test_evaluate_model(self):
        # Dummy model and data
        model = LogisticRegression()
        X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        y = np.array([0, 1, 1, 0])
        model.fit(X, y)
        metrics = evaluate_model(model, X, y)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)


if __name__ == '__main__':
    unittest.main()

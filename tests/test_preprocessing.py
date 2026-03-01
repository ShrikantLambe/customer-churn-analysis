import unittest
import pandas as pd
from src.preprocessing import load_data


class TestPreprocessing(unittest.TestCase):
    def test_load_data(self):
        # Test loading a small DataFrame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df.to_csv('tests/test.csv', index=False)
        loaded = load_data('tests/test.csv')
        self.assertEqual(loaded.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()

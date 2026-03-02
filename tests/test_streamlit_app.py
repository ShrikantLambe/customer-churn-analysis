import unittest


class TestStreamlitApp(unittest.TestCase):
    def test_app_file_exists(self):
        self.assertTrue(os.path.exists('app/streamlit_app.py'))


if __name__ == '__main__':
    unittest.main()

import json
import unittest

import app


class TestFlask(unittest.TestCase):

    def setUp(self):
        # Create a test client.
        self.app = app.app.test_client()
        # Propagate the exceptions to the test client.
        self.app.testing = True
        self.app.debug = True
        with open('example_request_data.json') as f:
            self.example_params = json.load(f)

    def test_get(self):
        response = self.app.get("api/v1/predict", data=self.example_params)
        self.assertEqual(response.status_code, 200)

    def test_content_type(self):
        response = self.app.get("api/v1/predict", data=self.example_params)
        self.assertEqual(response.headers['content-type'], 'application/json')

    def test_content(self):
        response = self.app.get("api/v1/predict", data=self.example_params)
        result = json.loads(response.data.decode('utf8'))
        self.assertEqual(sorted(result.keys()),
                         sorted(['label', 'probability', 'sample_uuid']))
        self.assertIsInstance(result['label'], float)
        self.assertIn(result['label'], [0., 1.])
        self.assertIsInstance(result['probability'], float)
        self.assertTrue(0 <= result['probability'] <= 1)


if __name__ == '__main__':
    unittest.main()

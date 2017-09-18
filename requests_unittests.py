import json
import requests
import unittest

HOST = "http://127.0.0.1"
PORT = 5000
END_POINT = 'api/v1/predict'
url = "{host}:{port}/{end_point}".format(host=HOST, port=PORT,
                                         end_point=END_POINT)


class TestRequests(unittest.TestCase):

    def setUp(self):
        with open('example_request_data.json') as f:
            self.example_params = json.load(f)

    def test_get(self):
        response = requests.get(url, params=self.example_params)
        self.assertEqual(response.status_code, 200)

    def test_content_type(self):
        response = requests.get(url, params=self.example_params)
        self.assertEqual(response.headers['content-type'], 'application/json')

    def test_content(self):
        response = requests.get(url, params=self.example_params)
        result = response.json()
        self.assertEqual(sorted(result.keys()),
                         sorted(['label', 'probability', 'sample_uuid']))
        self.assertIsInstance(result['label'], float)
        self.assertTrue(result['label'], [0., 1.])
        self.assertIsInstance(result['probability'], float)
        self.assertTrue(0 <= result['probability'] <= 1)


if __name__ == '__main__':
    print("Testing on: {url}".format(url=url))
    unittest.main()

import os
from fastapi.testclient import TestClient
from app import app
import io
import unittest
from PIL import Image

client = TestClient(app)

class TestPredictEndpoint(unittest.TestCase):
    def test_predict_endpoint(self):
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("prediction_uid", data)
        self.assertIn("detection_count", data)
        self.assertIn("labels", data)

if __name__ == "__main__":
    unittest.main()

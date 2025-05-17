import os
from fastapi.testclient import TestClient
from main import app  # make sure this is your actual FastAPI app file
import io

client = TestClient(app)

def test_predict_endpoint():
    # Create a dummy image in memory
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Send POST request
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200, response.text
    data = response.json()

    # You can assert based on expected structure
    assert "prediction_uid" in data
    assert "detection_count" in data
    assert "labels" in data

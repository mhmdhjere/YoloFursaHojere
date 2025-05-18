from fastapi.testclient import TestClient
from app import app, get_db
import sqlite3
import pytest

# ---- Test Setup ----
def override_get_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create tables
    conn.execute("""
    CREATE TABLE prediction_sessions (
        uid TEXT PRIMARY KEY,
        timestamp TEXT,
        original_image TEXT,
        predicted_image TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE detection_objects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_uid TEXT,
        label TEXT,
        score REAL,
        box TEXT
    )
    """)

    # Insert mock data
    conn.execute("""
    INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image)
    VALUES (?, ?, ?, ?)
    """, ("test-uid", "2024-01-01T12:00:00", "original.jpg", "predicted.jpg"))

    conn.execute("""
    INSERT INTO detection_objects (prediction_uid, label, score, box)
    VALUES (?, ?, ?, ?)
    """, ("test-uid", "car", 0.95, "[100, 200, 300, 400]"))

    return conn

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

# ---- Tests ----
def test_get_prediction_success():
    response = client.get("/prediction/test-uid")
    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "test-uid"
    assert data["original_image"] == "original.jpg"
    assert data["predicted_image"] == "predicted.jpg"
    assert len(data["detection_objects"]) == 1
    assert data["detection_objects"][0]["label"] == "car"

def test_get_prediction_not_found():
    response = client.get("/prediction/unknown-uid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"

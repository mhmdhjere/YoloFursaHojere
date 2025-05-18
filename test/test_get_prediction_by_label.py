import pytest
import sqlite3
from fastapi.testclient import TestClient
from app import app, get_db

# ---- Override DB ----
def override_get_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create schema
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
    conn.execute("INSERT INTO prediction_sessions VALUES (?, ?, ?, ?)",
                 ("abc123", "2024-01-01T10:00:00", "orig1.jpg", "pred1.jpg"))
    conn.execute("INSERT INTO prediction_sessions VALUES (?, ?, ?, ?)",
                 ("xyz789", "2024-01-02T11:00:00", "orig2.jpg", "pred2.jpg"))
    conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                 ("abc123", "cat", 0.9, "[10, 20, 30, 40]"))
    conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                 ("xyz789", "dog", 0.8, "[50, 60, 70, 80]"))
    conn.commit()
    return conn

# Inject the override
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

# ---- Tests ----

def test_get_predictions_by_label_cat():
    response = client.get("/predictions/label/cat")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["uid"] == "abc123"
    assert data[0]["timestamp"] == "2024-01-01T10:00:00"

def test_get_predictions_by_label_dog():
    response = client.get("/predictions/label/dog")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["uid"] == "xyz789"

def test_get_predictions_by_label_none():
    response = client.get("/predictions/label/elephant")
    assert response.status_code == 200
    data = response.json()
    assert data == []

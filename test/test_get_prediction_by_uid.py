import unittest
import sqlite3
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from app import app, get_prediction_by_uid  # Adjust `app` import as needed

# Create an in-memory SQLite DB and inject it
def override_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create schema
    with conn:
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

        # Add test data
        conn.execute("""
            INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image)
            VALUES ('test123', '2024-01-01 12:00:00', 'orig.jpg', 'pred.jpg')
        """)
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES ('test123', 'dog', 0.88, '[1,2,3,4]')
        """)

    return conn

# Modify the route to accept DB as dependency
# You need to make a small change in your route:
#
# def get_prediction_by_uid(uid: str, conn=Depends(get_db)):
#
# And define get_db() somewhere in your code:
#
# def get_db():
#     return sqlite3.connect(DB_PATH)
#
# This way, we can override it here:
app.dependency_overrides[get_prediction_by_uid.__globals__["sqlite3"].connect] = override_db

client = TestClient(app)

class TestGetPredictionAPI(unittest.TestCase):

    def test_successful_prediction_lookup(self):
        response = client.get("/prediction/test123")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], "test123")
        self.assertEqual(data["original_image"], "orig.jpg")
        self.assertEqual(data["detection_objects"][0]["label"], "dog")

    def test_prediction_not_found(self):
        response = client.get("/prediction/unknown123")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")


if __name__ == "__main__":
    unittest.main()

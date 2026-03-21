import sys
import os
from unittest.mock import MagicMock

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCKING STRATEGY:
# The `src.api.main` module imports `src.api.endpoints`, which in turn imports
# heavy libraries like `pandas`, `shap`, `torch`, etc.
# To test ONLY the new bundle endpoints without installing the entire data science stack,
# we will mock `src.api.endpoints` in sys.modules BEFORE importing `src.api.main`.

# 1. Create a mock for src.api.endpoints
mock_endpoints = MagicMock()
mock_endpoints.router = MagicMock() # The router object that gets included

# 2. Inject it into sys.modules
sys.modules['src.api.endpoints'] = mock_endpoints

# 3. NOW import the app
from src.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_analyze_endpoint_high_risk():
    payload = {
        "text": "The company is facing imminent bankruptcy. This is a severe issue [doc_1:chunk_1]."
    }
    response = client.post("/api/v1/bundle/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["risk_rating"] == "High Risk"
    assert "bankruptcy" in data["semantic_breakdown"][0]
    # 2 sentences ("...bankruptcy.", "...issue..."), 1 citation -> 0.5 score
    assert data["conviction_score"] == 0.5

def test_analyze_endpoint_low_risk():
    payload = {
        "text": "The company is performing well. Revenue is up [doc_2:chunk_5]. Margins are stable."
    }
    response = client.post("/api/v1/bundle/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["risk_rating"] == "Low Risk"
    assert len(data["semantic_breakdown"]) == 0
    # 3 sentences, 1 citation -> 1/3 = 0.333...
    assert abs(data["conviction_score"] - 0.333) < 0.01

def test_analyze_endpoint_no_text():
    response = client.post("/api/v1/bundle/analyze", json={"text": ""})
    assert response.status_code == 400

import os
import pytest
from datetime import datetime, timedelta
from src.risk_map.visualization import RiskVisualizationService

def test_risk_trajectory_generation(tmp_path):
    service = RiskVisualizationService()

    # Mock data
    history = [
        {"timestamp": datetime.now(), "company_id": "C1", "industry_sector": "Tech", "pd": 0.01},
        {"timestamp": datetime.now() + timedelta(hours=1), "company_id": "C1", "industry_sector": "Tech", "pd": 0.02},
        {"timestamp": datetime.now(), "company_id": "C2", "industry_sector": "Finance", "pd": 0.05},
        {"timestamp": datetime.now() + timedelta(hours=1), "company_id": "C2", "industry_sector": "Finance", "pd": 0.04},
    ]

    output_file = tmp_path / "test_plot.png"

    # Generate plot to file
    service.generate_risk_trajectory_plot(history, output_path=str(output_file))

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0

def test_risk_trajectory_generation_bytes():
    service = RiskVisualizationService()

    # Mock data
    history = [
        {"timestamp": datetime.now(), "company_id": "C1", "industry_sector": "Tech", "pd": 0.01},
    ]

    # Generate bytes
    img_bytes = service.generate_risk_trajectory_plot(history)

    assert img_bytes is not None
    assert len(img_bytes) > 0
    assert img_bytes.startswith(b'\x89PNG')

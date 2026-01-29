from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    # Mock model being loaded
    with patch("src.api.main.model", MagicMock()): 
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_health_check_unhealthy():
    # Mock model NOT being loaded
    with patch("src.api.main.model", None):
        response = client.get("/health")
        assert response.json() == {"status": "unhealthy", "reason": "Model not loaded"}

@patch("src.api.main.predict")
def test_predict_endpoint(mock_predict):
    # Mock prediction result
    mock_predict.return_value = ("dog", 0.95)
    
    # Mock model loaded
    with patch("src.api.main.model", MagicMock()):
        # Create a dummy image content
        files = {'file': ('test.jpg', b'fakeimagebytes', 'image/jpeg')}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["prediction"] == "dog"
        assert json_response["confidence"] == "0.95"

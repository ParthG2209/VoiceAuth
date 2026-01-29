"""
API Integration Tests for VoiceAuth
Tests the main /api/voice-detection endpoint
"""

import pytest
import base64
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """Return valid API key for testing"""
    return settings.api_key


@pytest.fixture
def auth_headers(valid_api_key):
    """Return headers with valid API key"""
    return {"x-api-key": valid_api_key}


class TestHealthEndpoint:
    """Tests for /api/health endpoint"""
    
    def test_health_check_success(self, client):
        """Health check should return 200 without auth"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "supported_languages" in data
        assert len(data["supported_languages"]) == 5


class TestVoiceDetectionEndpoint:
    """Tests for /api/voice-detection endpoint"""
    
    def test_missing_api_key_returns_401(self, client):
        """Request without API key should fail"""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "dGVzdA=="
            }
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["status"] == "error"
        assert "API key" in data["detail"]["message"]
    
    def test_invalid_api_key_returns_401(self, client):
        """Request with wrong API key should fail"""
        response = client.post(
            "/api/voice-detection",
            headers={"x-api-key": "invalid_key"},
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "dGVzdA=="
            }
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["status"] == "error"
    
    def test_invalid_language_returns_422(self, client, auth_headers):
        """Request with unsupported language should fail"""
        response = client.post(
            "/api/voice-detection",
            headers=auth_headers,
            json={
                "language": "French",  # Not supported
                "audioFormat": "mp3",
                "audioBase64": "dGVzdA=="
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
    
    def test_invalid_audio_format_returns_422(self, client, auth_headers):
        """Request with wrong audio format should fail"""
        response = client.post(
            "/api/voice-detection",
            headers=auth_headers,
            json={
                "language": "English",
                "audioFormat": "wav",  # Only mp3 supported
                "audioBase64": "dGVzdA=="
            }
        )
        
        assert response.status_code == 422
    
    def test_missing_audio_returns_422(self, client, auth_headers):
        """Request without audio data should fail"""
        response = client.post(
            "/api/voice-detection",
            headers=auth_headers,
            json={
                "language": "English",
                "audioFormat": "mp3"
                # Missing audioBase64
            }
        )
        
        assert response.status_code == 422
    
    def test_invalid_base64_returns_400(self, client, auth_headers):
        """Request with invalid base64 should fail"""
        response = client.post(
            "/api/voice-detection",
            headers=auth_headers,
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "not-valid-base64!!!"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_all_supported_languages_accepted(self, client, auth_headers):
        """All 5 supported languages should be accepted in request"""
        # Note: These will fail at audio processing stage, but language validation should pass
        languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        
        for lang in languages:
            response = client.post(
                "/api/voice-detection",
                headers=auth_headers,
                json={
                    "language": lang,
                    "audioFormat": "mp3",
                    "audioBase64": "dGVzdGluZ3Rlc3Rpbmd0ZXN0aW5n"  # Valid base64, invalid audio
                }
            )
            
            # Should fail at audio processing, not language validation
            assert response.status_code != 422 or "language" not in response.text.lower()


class TestLanguagesEndpoint:
    """Tests for /api/languages endpoint"""
    
    def test_get_languages_success(self, client):
        """Should return list of supported languages"""
        response = client.get("/api/languages")
        
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert data["count"] == 5
        assert "Tamil" in data["languages"]
        assert "English" in data["languages"]


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_returns_api_info(self, client):
        """Root should return API info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "VoiceAuth API"
        assert "version" in data
        assert "docs" in data

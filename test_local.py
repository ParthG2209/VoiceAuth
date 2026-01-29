"""
Test Script for VoiceAuth API
Tests the API with generated audio samples
"""

import base64
import json
import time
from pathlib import Path

import httpx
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment


# Configuration
API_URL = "http://localhost:8000"
API_KEY = "sk_voiceauth_dev_key_12345"  # Update with your key


def generate_test_audio(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a simple test audio (sine wave)
    This simulates a basic audio sample for testing
    """
    print(f"   Generating {duration_sec}s test audio...")
    
    # Generate sine wave (440 Hz - A note)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    frequency = 440.0
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more realistic
    audio_data += 0.1 * np.sin(2 * np.pi * 880 * t)  # Harmonic
    audio_data += 0.05 * np.random.randn(len(audio_data))  # Noise
    
    # Normalize to 16-bit range
    audio_data = np.int16(audio_data * 32767 * 0.8)
    
    # Save as WAV temporarily
    temp_wav = Path("temp_test.wav")
    wavfile.write(temp_wav, sample_rate, audio_data)
    
    # Convert to MP3
    audio = AudioSegment.from_wav(str(temp_wav))
    temp_mp3 = Path("temp_test.mp3")
    audio.export(str(temp_mp3), format="mp3")
    
    # Read MP3 bytes
    with open(temp_mp3, "rb") as f:
        mp3_bytes = f.read()
    
    # Cleanup
    temp_wav.unlink()
    temp_mp3.unlink()
    
    return mp3_bytes


def test_health_check():
    """Test the health check endpoint"""
    print("\n1️⃣  Testing Health Check...")
    
    try:
        response = httpx.get(f"{API_URL}/api/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"   ✅ Status: {data['status']}")
        print(f"   📌 Version: {data['version']}")
        print(f"   🌍 Languages: {', '.join(data['supported_languages'])}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return False


def test_voice_detection(language: str = "English", use_deep_learning: bool = False):
    """Test voice detection endpoint"""
    print(f"\n2️⃣  Testing Voice Detection ({language})...")
    print(f"   Deep Learning: {use_deep_learning}")
    
    try:
        # Generate test audio
        mp3_bytes = generate_test_audio(duration_sec=2.0)
        audio_base64 = base64.b64encode(mp3_bytes).decode()
        
        print(f"   📊 Audio size: {len(mp3_bytes)} bytes")
        print(f"   📊 Base64 size: {len(audio_base64)} chars")
        
        # Make API request
        print("   🔄 Sending request to API...")
        start_time = time.time()
        
        response = httpx.post(
            f"{API_URL}/api/voice-detection",
            params={"use_deep_learning": use_deep_learning},
            headers={
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            },
            timeout=60.0  # Longer timeout for first Wav2Vec2 download
        )
        
        elapsed = time.time() - start_time
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n   ✅ Response received in {elapsed:.2f}s")
        print(f"   📋 Classification: {result['classification']}")
        print(f"   📊 Confidence: {result['confidenceScore']:.2%}")
        print(f"   💬 Explanation: {result['explanation']}")
        
        return True
        
    except httpx.HTTPStatusError as e:
        print(f"   ❌ HTTP Error: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return False


def test_authentication():
    """Test API key authentication"""
    print("\n3️⃣  Testing Authentication...")
    
    try:
        # Test without API key
        print("   Testing without API key...")
        response = httpx.post(
            f"{API_URL}/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "dGVzdA=="
            }
        )
        
        if response.status_code == 401:
            print("   ✅ Correctly rejected request without API key")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
        
        # Test with invalid API key
        print("   Testing with invalid API key...")
        response = httpx.post(
            f"{API_URL}/api/voice-detection",
            headers={"x-api-key": "invalid_key"},
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "dGVzdA=="
            }
        )
        
        if response.status_code == 401:
            print("   ✅ Correctly rejected invalid API key")
            return True
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return False


def test_all_languages():
    """Test all supported languages"""
    print("\n4️⃣  Testing All Languages...")
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    for lang in languages:
        print(f"\n   Testing {lang}...")
        try:
            mp3_bytes = generate_test_audio(duration_sec=1.0)
            audio_base64 = base64.b64encode(mp3_bytes).decode()
            
            response = httpx.post(
                f"{API_URL}/api/voice-detection",
                params={"use_deep_learning": False},  # Faster for testing
                headers={"x-api-key": API_KEY},
                json={
                    "language": lang,
                    "audioFormat": "mp3",
                    "audioBase64": audio_base64
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {lang}: {result['classification']} ({result['confidenceScore']:.2%})")
            else:
                print(f"   ❌ {lang}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {lang}: {str(e)}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("🎙️  VoiceAuth API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        httpx.get(API_URL, timeout=2.0)
    except Exception:
        print("\n❌ Server is not running!")
        print("   Please start the server first:")
        print("   uvicorn app.main:app --reload --port 8000")
        return
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("Authentication", test_authentication()))
    results.append(("Voice Detection (Feature-based)", test_voice_detection("English", False)))
    results.append(("Voice Detection (Deep Learning)", test_voice_detection("English", True)))
    results.append(("All Languages", test_all_languages()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()

import base64
import requests
import sys

# CONFIGURATION
API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_voiceauth_dev_key_12345"

def test_file(file_path):
    print(f"Processing: {file_path}")
    
    # 1. Read and Encode File
    try:
        with open(file_path, "rb") as f:
            audio_content = f.read()
            encoded_string = base64.b64encode(audio_content).decode('utf-8')
    except FileNotFoundError:
        print("Error: File not found!")
        return

    # 2. Send Request
    payload = {
        "language": "English",  # Change if needed
        "audioFormat": "mp3",   # or "wav"
        "audioBase64": encoded_string
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("\n----- RESULT -----")
            print(f"Class: {result['classification']}")
            print(f"Conf : {result['confidenceScore']:.2%}")
            print(f"Note : {result['explanation']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_custom.py <path_to_audio_file>")
    else:
        test_file(sys.argv[1])
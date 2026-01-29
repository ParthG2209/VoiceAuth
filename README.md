# 🎙️ VoiceAuth - AI Voice Detection API

Detect whether a voice sample is **AI-generated** or **Human** across 5 languages.

## 🌍 Supported Languages
- Tamil
- English  
- Hindi
- Malayalam
- Telugu

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- FFmpeg (for audio processing)

### Installation

```bash
# Clone and navigate
cd VoiceAuth

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API key
```

### Run the Server

```bash
# Development mode
uvicorn app.main:app --reload --port 8000

# Or directly
python -m app.main
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 API Usage

### Endpoint
```
POST /api/voice-detection
```

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Request Body
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3..."
}
```

### Response
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency detected"
}
```

### cURL Example
```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_voiceauth_dev_key_12345" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_AUDIO"
  }'
```

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# Or just build
docker build -t voiceauth .
docker run -p 8000:8000 voiceauth
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📁 Project Structure
```
VoiceAuth/
├── app/
│   ├── api/
│   │   ├── auth.py       # API key validation
│   │   ├── routes.py     # Endpoints
│   │   └── schemas.py    # Request/Response models
│   ├── models/
│   │   └── detector.py   # AI detection logic
│   ├── utils/
│   │   └── audio_processor.py
│   ├── config.py
│   └── main.py           # FastAPI app
├── tests/
├── Dockerfile
└── requirements.txt
```

## 🔒 Security
- All endpoints require `x-api-key` header
- Set your API key in `.env` file

## 📝 License
MIT License

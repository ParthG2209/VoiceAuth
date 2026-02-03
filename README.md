# 🎙️ VoiceAuth - AI Voice Detection API

Detect whether a voice sample is **AI-generated** or **Human** across 5 languages using a custom-trained deep learning model.

## 🌍 Supported Languages
- Tamil
- English  
- Hindi
- Malayalam
- Telugu

## 🧠 Detection Method

Uses a **custom-trained Keras model** specifically designed to classify AI-generated vs human voices:
- Trained on a diverse dataset of AI and human voice samples
- Extracts comprehensive audio features (MFCC, spectral, energy patterns)
- High accuracy classification with confidence scoring

---

## 🚀 Quick Start (Local Testing)

### Prerequisites
- Python 3.10+ 
- FFmpeg (for audio processing)

### Option 1: Automated Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/ParthG2209/VoiceAuth.git
cd VoiceAuth

# 2. Run setup script
./setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Start the server
./run.sh
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create directories
mkdir -p data/sample_audio logs

# 4. Set up environment
cp .env.example .env
# Edit .env and update your API_KEY

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

---

## 📡 API Usage

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/api/health
```

#### 2. Voice Detection
```bash
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_voiceauth_dev_key_12345" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_ENCODED_MP3"
  }'
```

**Response:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Custom model detected AI-generated voice patterns (confidence: 87.0%)"
}
```

---

## 🧪 Testing

### Run Test Suite
```bash
# Make sure server is running first
source venv/bin/activate
python test_local.py
```

This will test:
- ✅ Health check endpoint
- ✅ API authentication
- ✅ Voice detection
- ✅ All 5 supported languages

### Run Unit Tests
```bash
pytest tests/ -v
```

---

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🔑 Authentication

All endpoints (except `/health`) require the `x-api-key` header:

```bash
x-api-key: YOUR_API_KEY
```

Set your API key in `.env`:
```bash
API_KEY=sk_voiceauth_your_secret_key_here
```

---

## 📁 Project Structure

```
VoiceAuth/
├── app/
│   ├── api/
│   │   ├── auth.py              # API key validation
│   │   ├── routes.py            # API endpoints
│   │   └── schemas.py           # Request/Response models
│   ├── models/
│   │   ├── custom_classifier.py # Custom trained Keras model
│   │   ├── detector.py          # Feature analysis module
│   │   └── ensemble.py          # Detection orchestrator
│   ├── utils/
│   │   └── audio_processor.py   # Audio processing pipeline
│   ├── config.py                # Configuration
│   └── main.py                  # FastAPI app
├── new_model/                   # Custom trained model files
│   ├── ai_human_voice_classifier.h5
│   ├── feature_scaler.pkl
│   └── label_encoder.pkl
├── tests/
│   └── test_api.py              # API tests
├── setup.sh                     # Setup script
├── run.sh                       # Run server script
├── test_local.py                # Local testing script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or with Docker directly
docker build -t voiceauth .
docker run -p 8000:8000 -e API_KEY=your_key voiceauth
```

---

## ⚙️ Configuration

Edit `.env` file:

```bash
# API Security
API_KEY=sk_voiceauth_your_secret_key_here

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Models
MODEL_CACHE_DIR=./new_model
USE_GPU=false

# Audio Limits
MAX_AUDIO_SIZE_MB=10
MAX_AUDIO_DURATION_SECONDS=60
```

---

## 📊 Model Performance

### Response Times
- Typical request: ~1-2 seconds
- First request may take slightly longer (model loading)

### Model Details
- **Architecture:** Custom Keras neural network
- **Input:** Audio features (MFCC, spectral, energy patterns)
- **Output:** Binary classification (AI_GENERATED / HUMAN) with confidence score

---

## 🔧 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use a different port
uvicorn app.main:app --port 8080
```

### Dependencies installation fails
```bash
# Install system dependencies (macOS)
brew install ffmpeg

# Install system dependencies (Ubuntu)
sudo apt-get install ffmpeg libsndfile1
```

### Model loading fails
```bash
# Ensure the new_model directory exists with all files
ls -la new_model/
# Should show:
# - ai_human_voice_classifier.h5
# - feature_scaler.pkl
# - label_encoder.pkl
```

---

## 🎯 Supported Audio Formats

- **Input:** MP3 (Base64 encoded)
- **Sample Rate:** Automatically resampled to 16kHz
- **Max Duration:** 60 seconds (configurable)
- **Max Size:** 10MB (configurable)

---

## 📝 Example: Convert Audio to Base64

### Python
```python
import base64

with open("audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()
```

### Command Line
```bash
base64 -i audio.mp3 -o audio.txt
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Links

- **GitHub:** https://github.com/ParthG2209/VoiceAuth
- **API Docs:** http://localhost:8000/docs (when running)

---

## 💡 Tips

1. Use Swagger UI for interactive API testing
2. Check logs in console for detailed processing info
3. The custom model is optimized for common AI voice generators

---

**Built with ❤️ using FastAPI, TensorFlow/Keras, and Librosa**

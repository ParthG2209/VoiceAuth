# 🎙️ VoiceAuth - AI Voice Detection API

Detect whether a voice sample is **AI-generated** or **Human** across 5 languages using ensemble ML models.

## 🌍 Supported Languages
- Tamil
- English  
- Hindi
- Malayalam
- Telugu

## 🧠 Detection Methods
1. **Feature-based Analysis** - MFCC, pitch, spectral, energy patterns
2. **Wav2Vec2 Deep Learning** - Facebook's pre-trained speech model
3. **Ensemble Voting** - Combines both methods for best accuracy

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
mkdir -p models data/sample_audio logs

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
curl -X POST "http://localhost:8000/api/voice-detection?use_deep_learning=true" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_voiceauth_dev_key_12345" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_ENCODED_MP3"
  }'
```

**Query Parameters:**
- `use_deep_learning` (optional, default: `true`)
  - `true` - Use Wav2Vec2 + features (more accurate, slower first time)
  - `false` - Use features only (faster)

**Response:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Low temporal variation in speech embeddings; High frame-to-frame consistency"
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
- ✅ Voice detection (feature-based)
- ✅ Voice detection (deep learning)
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
│   │   ├── detector.py          # Feature-based detector
│   │   ├── wav2vec2_detector.py # Deep learning detector
│   │   └── ensemble.py          # Ensemble combining both
│   ├── utils/
│   │   └── audio_processor.py   # Audio processing pipeline
│   ├── config.py                # Configuration
│   └── main.py                  # FastAPI app
├── tests/
│   └── test_api.py              # API tests
├── models/                      # Downloaded ML models (auto-created)
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
MODEL_CACHE_DIR=./models
USE_GPU=false

# Audio Limits
MAX_AUDIO_SIZE_MB=10
MAX_AUDIO_DURATION_SECONDS=60
```

---

## 📊 Model Performance

### First Request (Wav2Vec2 Download)
- Downloads ~360MB model from Hugging Face
- Takes 1-2 minutes (one-time only)
- Cached in `./models/` for future use

### Subsequent Requests
- Feature-based: ~0.5-1s
- With Wav2Vec2: ~2-3s
- Ensemble: ~2-3s

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

### Wav2Vec2 download fails
```bash
# Set Hugging Face cache directory
export HF_HOME=./models
export TRANSFORMERS_CACHE=./models
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

1. **First run with Wav2Vec2** will download the model (~360MB)
2. Use `use_deep_learning=false` for faster testing
3. Check logs in console for detailed processing info
4. Use Swagger UI for interactive API testing

---

**Built with ❤️ using FastAPI, Hugging Face Transformers, and Librosa**

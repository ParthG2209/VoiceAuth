#!/bin/bash

# VoiceAuth - Local Setup Script
# This script sets up the virtual environment and installs all dependencies

set -e  # Exit on error

echo "🎙️  VoiceAuth - Local Setup"
echo "=============================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data/sample_audio
mkdir -p logs

# Copy .env file if it doesn't exist
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✅ Created .env file from .env.example"
    echo "   ⚠️  Please update your API key in .env"
else
    echo "   .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Activate the virtual environment:"
echo "      source venv/bin/activate"
echo ""
echo "   2. Update your API key in .env file"
echo ""
echo "   3. Run the server:"
echo "      uvicorn app.main:app --reload --port 8000"
echo ""
echo "   4. Visit the API docs:"
echo "      http://localhost:8000/docs"
echo ""

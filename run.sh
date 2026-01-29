#!/bin/bash

# VoiceAuth - Run Server Script
# Quick script to start the development server

set -e

echo "🎙️  Starting VoiceAuth API Server..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run ./setup.sh first"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found, using defaults"
fi

# Run server
echo "🚀 Starting server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

#!/bin/bash
# ==============================================================================
# VESPER Virtual Assistant - Setup Script
# ==============================================================================
# This script installs all dependencies for the VESPER voice assistant.
# 
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# ==============================================================================

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           VESPER Virtual Assistant - Setup                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📌 Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install core dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install optional voice dependencies
echo ""
echo "📥 Installing voice dependencies..."
pip install vosk pyttsx3 pyaudio SpeechRecognition

# Create models directory
echo ""
echo "📁 Creating models directory..."
mkdir -p models

# Check if Vosk model exists
VOSK_MODEL="models/vosk-model-small-en-us-0.15"
if [ ! -d "$VOSK_MODEL" ]; then
    echo ""
    echo "⚠️  Vosk model not found!"
    echo "   Download from: https://alphacephei.com/vosk/models"
    echo "   Recommended: vosk-model-small-en-us-0.15"
    echo ""
    echo "   Run these commands:"
    echo "   cd models"
    echo "   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    echo "   unzip vosk-model-small-en-us-0.15.zip"
    echo ""
fi

# Check if whisper.cpp exists
WHISPER_BIN="/usr/local/bin/whisper-cpp"
if [ ! -f "$WHISPER_BIN" ]; then
    echo "⚠️  whisper.cpp not found!"
    echo "   Build from: https://github.com/ggerganov/whisper.cpp"
    echo ""
    echo "   Run these commands:"
    echo "   git clone https://github.com/ggerganov/whisper.cpp"
    echo "   cd whisper.cpp"
    echo "   make"
    echo "   sudo cp main /usr/local/bin/whisper-cpp"
    echo "   ./models/download-ggml-model.sh base.en"
    echo "   cp models/ggml-base.en.bin ../models/"
    echo ""
fi

# Check for PyAudio dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "⚠️  Homebrew not found. Install from: https://brew.sh"
    else
        if ! brew list portaudio &> /dev/null; then
            echo "📥 Installing PortAudio for microphone access..."
            brew install portaudio
        fi
    fi
fi

# Create data directory
mkdir -p data logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run VESPER:"
echo "  source venv/bin/activate"
echo "  export GEMINI_API_KEY='your-api-key'"
echo "  python3 main.py"
echo ""

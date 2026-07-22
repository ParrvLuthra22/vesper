#!/bin/bash
# =============================================================================
# whisper.cpp Installation Script for macOS
# =============================================================================
# This script:
#   1. Clones whisper.cpp from GitHub
#   2. Builds it with optimizations for Apple Silicon / Intel
#   3. Downloads a speech recognition model
#   4. Installs the binary to /usr/local/bin
#
# Usage:
#   chmod +x scripts/install_whisper_cpp.sh
#   ./scripts/install_whisper_cpp.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           whisper.cpp Installation for VESPER                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Configuration
WHISPER_DIR="${HOME}/.local/whisper.cpp"
MODEL_DIR="$(pwd)/models"
MODEL_NAME="base.en"  # Options: tiny, tiny.en, base, base.en, small, small.en, medium, large

# Check for required tools
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    echo "Install with: brew install git"
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo -e "${RED}Error: make is not installed${NC}"
    echo "Install Xcode Command Line Tools with: xcode-select --install"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Clone whisper.cpp
echo -e "${YELLOW}[2/5] Cloning whisper.cpp...${NC}"

if [ -d "$WHISPER_DIR" ]; then
    echo "whisper.cpp already exists at $WHISPER_DIR"
    echo "Updating..."
    cd "$WHISPER_DIR"
    git pull
else
    echo "Cloning to $WHISPER_DIR..."
    git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    cd "$WHISPER_DIR"
fi

echo -e "${GREEN}✓ Repository ready${NC}"

# Build whisper.cpp
echo -e "${YELLOW}[3/5] Building whisper.cpp...${NC}"

# Clean any previous build
make clean 2>/dev/null || true

# Detect architecture and build with appropriate flags
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "Detected Apple Silicon (M1/M2/M3)"
    # Build with Metal acceleration for Apple Silicon
    make -j$(sysctl -n hw.ncpu) WHISPER_METAL=1
else
    echo "Detected Intel Mac"
    # Build with standard optimizations
    make -j$(sysctl -n hw.ncpu)
fi

echo -e "${GREEN}✓ Build complete${NC}"

# Download model
echo -e "${YELLOW}[4/5] Downloading model (${MODEL_NAME})...${NC}"

mkdir -p "$MODEL_DIR"

# Download the model using the provided script
bash ./models/download-ggml-model.sh "$MODEL_NAME"

# Copy model to our models directory
MODEL_FILE="models/ggml-${MODEL_NAME}.bin"
if [ -f "$MODEL_FILE" ]; then
    cp "$MODEL_FILE" "$MODEL_DIR/"
    echo -e "${GREEN}✓ Model downloaded to ${MODEL_DIR}/ggml-${MODEL_NAME}.bin${NC}"
else
    echo -e "${RED}Warning: Model file not found at expected location${NC}"
    echo "You may need to download manually"
fi

# Install binary
echo -e "${YELLOW}[5/5] Installing whisper-cpp binary...${NC}"

# Create /usr/local/bin if it doesn't exist
if [ ! -d "/usr/local/bin" ]; then
    echo "Creating /usr/local/bin (requires sudo)..."
    sudo mkdir -p /usr/local/bin
fi

# Copy the main binary
echo "Installing to /usr/local/bin/whisper-cpp (requires sudo)..."
sudo cp main /usr/local/bin/whisper-cpp
sudo chmod +x /usr/local/bin/whisper-cpp

echo -e "${GREEN}✓ Installed to /usr/local/bin/whisper-cpp${NC}"

# Verify installation
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "whisper.cpp binary: /usr/local/bin/whisper-cpp"
echo "Model file: ${MODEL_DIR}/ggml-${MODEL_NAME}.bin"
echo ""
echo "Test with:"
echo "  whisper-cpp --help"
echo ""
echo "Or transcribe audio:"
echo "  whisper-cpp -m ${MODEL_DIR}/ggml-${MODEL_NAME}.bin -f audio.wav"
echo ""

# Quick test
echo -e "${YELLOW}Running quick test...${NC}"
if whisper-cpp --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓ whisper-cpp is working!${NC}"
else
    echo -e "${RED}Warning: whisper-cpp test failed${NC}"
    echo "You may need to add /usr/local/bin to your PATH"
fi

#!/bin/bash
# Download a tiny GGUF model for testing hemllama bindings
#
# This downloads a ~15MB "stories" model that's good enough
# to test tokenization, decoding, and generation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
MODEL_FILE="$MODELS_DIR/tinyllama-stories-15m-q4_0.gguf"

# TinyStories 15M model quantized to Q4_0 (~15MB)
MODEL_URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo "Test model already exists: $MODEL_FILE"
    echo "Size: $(du -h "$MODEL_FILE" | cut -f1)"
    exit 0
fi

echo "Downloading tiny test model (~15MB)..."
echo "Source: $MODEL_URL"
echo ""

mkdir -p "$MODELS_DIR"

if command -v curl &> /dev/null; then
    curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
elif command -v wget &> /dev/null; then
    wget -O "$MODEL_FILE" "$MODEL_URL" --show-progress
else
    echo "Error: curl or wget required"
    exit 1
fi

echo ""
echo "Downloaded: $MODEL_FILE"
echo "Size: $(du -h "$MODEL_FILE" | cut -f1)"
echo ""
echo "Run tests with:"
echo "  hemlock test_model.hml $MODEL_FILE"
echo ""
echo "Or try the examples:"
echo "  hemlock examples/simple.hml $MODEL_FILE \"Once upon a time\""
echo "  hemlock examples/model_info.hml $MODEL_FILE"
echo "  hemlock examples/tokenizer.hml $MODEL_FILE"

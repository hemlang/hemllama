#!/bin/bash
# Build script for hemllama - builds llama.cpp as a shared library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
LIB_DIR="$SCRIPT_DIR/lib"

echo "Building llama.cpp..."

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake - build shared library
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    "$@"

# Build
cmake --build . --config Release -j$(nproc)

# Create lib directory and copy the shared library
mkdir -p "$LIB_DIR"

# Find and copy the shared library (different names on different platforms)
if [ -f "$BUILD_DIR/src/libllama.so" ]; then
    cp "$BUILD_DIR/src/libllama.so"* "$LIB_DIR/"
    cp "$BUILD_DIR/ggml/src/libggml.so"* "$LIB_DIR/" 2>/dev/null || true
    cp "$BUILD_DIR/ggml/src/libggml-base.so"* "$LIB_DIR/" 2>/dev/null || true
    cp "$BUILD_DIR/ggml/src/libggml-cpu.so"* "$LIB_DIR/" 2>/dev/null || true
elif [ -f "$BUILD_DIR/src/libllama.dylib" ]; then
    cp "$BUILD_DIR/src/libllama.dylib" "$LIB_DIR/"
    cp "$BUILD_DIR/ggml/src/libggml"*.dylib "$LIB_DIR/" 2>/dev/null || true
fi

echo ""
echo "Build complete! Libraries installed to: $LIB_DIR"
echo ""
echo "To use hemllama, make sure the library path is set:"
echo "  export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\""

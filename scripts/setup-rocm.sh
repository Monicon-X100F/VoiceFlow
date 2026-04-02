#!/usr/bin/env bash
# VoiceFlow ROCm Setup Script
# Sets up AMD GPU (ROCm) support for VoiceFlow on Linux

set -euo pipefail

echo "============================================"
echo " VoiceFlow ROCm Setup"
echo "============================================"
echo ""

# Step 1: Check for AMD GPU
echo "--- Step 1: Checking for AMD GPU ---"
if ! lspci | grep -iq "VGA.*AMD\|Display.*AMD"; then
    echo "ERROR: No AMD GPU detected via lspci."
    echo "This script is only for systems with AMD GPUs."
    exit 1
fi
echo "AMD GPU detected."
echo ""

# Step 2: Check ROCm installation
echo "--- Step 2: Checking ROCm installation ---"
if ! command -v rocm-smi &>/dev/null; then
    echo "ERROR: rocm-smi not found. ROCm does not appear to be installed."
    echo ""
    echo "Please install ROCm first:"
    echo "  Official docs: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    echo ""
    echo "  Quick install (Ubuntu/Debian):"
    echo "    sudo apt install rocm-hip-sdk rocblas-dev hipblas-dev"
    echo ""
    exit 1
fi
echo "ROCm is installed (rocm-smi found)."
echo ""

# Step 3: Show detected GPU
echo "--- Step 3: Detected GPU ---"
rocm-smi --showproductname
echo ""

# Step 4: Check ROCm libraries
echo "--- Step 4: Checking ROCm libraries ---"
MISSING_LIBS=0
check_lib() {
    local lib="$1"
    if ldconfig -p | grep -q "$lib"; then
        echo "  [OK] $lib"
    else
        echo "  [X]  $lib  (NOT FOUND)"
        MISSING_LIBS=1
    fi
}

check_lib "libamdhip64.so"
check_lib "librocblas.so"
check_lib "libhipblas.so"

if [ "$MISSING_LIBS" -ne 0 ]; then
    echo ""
    echo "ERROR: One or more required ROCm libraries are missing."
    echo "Please ensure rocm-hip-sdk, rocblas-dev, and hipblas-dev are installed."
    exit 1
fi
echo "All required ROCm libraries found."
echo ""

# Step 5: Build and install ctranslate2 with ROCm
echo "--- Step 5: Building ctranslate2 with ROCm support ---"
echo "NOTE: This step builds ctranslate2 from source."
echo "      It requires: cmake, python3 development headers, and a C++ compiler."
echo ""

# Create temp directory
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Clone ROCm fork
echo "Cloning CTranslate2-rocm fork..."
git clone https://github.com/Yhapcele/CTranslate2-rocm "$TMPDIR/ctranslate2-rocm"

# Auto-detect GPU architecture
echo "Detecting GPU architecture..."
GFX_ARCH="$(rocminfo | grep -oP 'gfx\d+' | head -1 || true)"
if [ -z "$GFX_ARCH" ]; then
    GFX_ARCH="gfx1100"
    echo "Could not auto-detect GPU architecture, defaulting to $GFX_ARCH"
else
    echo "Detected GPU architecture: $GFX_ARCH"
fi

# Build
echo "Building ctranslate2 (this may take several minutes)..."
cmake -S "$TMPDIR/ctranslate2-rocm" \
      -B "$TMPDIR/ctranslate2-rocm/build" \
      -DWITH_HIP=ON \
      "-DCMAKE_HIP_ARCHITECTURES=$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release
make -C "$TMPDIR/ctranslate2-rocm/build" -j"$(nproc)"

# Install Python package
echo "Installing ctranslate2 Python package..."
pip install "$TMPDIR/ctranslate2-rocm/python/"

# Cleanup is handled by the trap above
echo ""
echo "============================================"
echo " ROCm setup complete!"
echo " Please restart VoiceFlow to apply changes."
echo "============================================"

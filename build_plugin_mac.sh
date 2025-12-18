#!/bin/bash
################################################################################
# CinematicFX Plugin Build Script for macOS
################################################################################

set -e  # Exit on error

echo "========================================"
echo "CinematicFX Plugin Build (macOS)"
echo "========================================"
echo ""

# Check for Xcode Command Line Tools
if ! xcode-select -p &> /dev/null; then
    echo "[ERROR] Xcode Command Line Tools not installed"
    echo "Please run: xcode-select --install"
    exit 1
fi
echo "[OK] Xcode Command Line Tools found"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "[ERROR] CMake not installed"
    echo "Please run: brew install cmake"
    exit 1
fi
echo "[OK] CMake found: $(cmake --version | head -n1)"

# Check for Adobe SDK
AE_SDK_PATH="$HOME/Downloads/AfterEffectsSDK_25.6_61_mac"
if [ ! -d "$AE_SDK_PATH" ]; then
    echo "[WARNING] Adobe SDK not found at: $AE_SDK_PATH"
    echo "Please download and extract the Adobe After Effects SDK"
    echo "Expected path: ~/Downloads/AfterEffectsSDK_25.6_61_mac/"
fi

echo ""
echo "Cleaning previous build..."
rm -rf build/
mkdir -p build

echo "Running CMake configuration..."
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCINEMATICFX_BUILD_METAL=ON \
      -DCINEMATICFX_BUILD_CUDA=OFF \
      ..

if [ $? -ne 0 ]; then
    echo "[ERROR] CMake configuration failed"
    exit 1
fi

echo ""
echo "Building plugin..."
make -j$(sysctl -n hw.ncpu)

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""

if [ -e "CinematicFX.plugin" ]; then
    echo "Plugin created: build/CinematicFX.plugin"
    echo "Size: $(du -h CinematicFX.plugin | cut -f1)"
    echo ""
    echo "Install locations:"
    echo "  Premiere Pro: /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
    echo "  After Effects: /Applications/Adobe After Effects 2025/Plug-ins/"
    echo ""
    echo "To install, run: ./install_mac.sh"
else
    echo "[ERROR] Plugin file not found after build"
    exit 1
fi

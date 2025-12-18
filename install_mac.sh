#!/bin/bash
################################################################################
# CinematicFX Plugin Installer for macOS
################################################################################

set -e

echo "========================================"
echo "CinematicFX Plugin Installer (macOS)"
echo "========================================"
echo ""

# Check if plugin exists
if [ ! -e "build/CinematicFX.plugin" ]; then
    echo "[ERROR] Plugin not found. Please build first:"
    echo "  ./build_plugin_mac.sh"
    exit 1
fi

# Installation paths
PREMIERE_PATH="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
AE_PATH="/Applications/Adobe After Effects 2025/Plug-ins"
USER_PREMIERE_PATH="$HOME/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"

INSTALLED=0

echo "Installing CinematicFX.plugin..."
echo ""

# Try Premiere Pro (system)
if [ -d "$PREMIERE_PATH" ]; then
    echo "[1/3] Installing to Premiere Pro (system)..."
    sudo cp -R "build/CinematicFX.plugin" "$PREMIERE_PATH/" && \
    echo "      ✓ Installed to: $PREMIERE_PATH" && \
    INSTALLED=1
fi

# Try After Effects
if [ -d "$AE_PATH" ]; then
    echo "[2/3] Installing to After Effects..."
    sudo cp -R "build/CinematicFX.plugin" "$AE_PATH/" && \
    echo "      ✓ Installed to: $AE_PATH" && \
    INSTALLED=1
fi

# Try user-specific Premiere
if [ -d "$USER_PREMIERE_PATH" ]; then
    echo "[3/3] Installing to Premiere Pro (user)..."
    cp -R "build/CinematicFX.plugin" "$USER_PREMIERE_PATH/" && \
    echo "      ✓ Installed to: $USER_PREMIERE_PATH" && \
    INSTALLED=1
fi

echo ""
if [ $INSTALLED -eq 1 ]; then
    echo "========================================"
    echo "  ✓ Installation Successful!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Restart Premiere Pro or After Effects"
    echo "  2. Go to: Effects → Video Effects → CinematicFX"
    echo "  3. Drag 'CinematicFX' onto your clip"
    echo "  4. Adjust parameters in Effect Controls"
    echo ""
else
    echo "========================================"
    echo "  ✗ Installation Failed"
    echo "========================================"
    echo ""
    echo "Adobe applications not found."
    echo "Manual installation:"
    echo "  1. Locate your Adobe app folder"
    echo "  2. Copy build/CinematicFX.plugin to:"
    echo "     - Premiere: /Library/.../Plug-ins/7.0/MediaCore/"
    echo "     - AE: /Applications/Adobe After Effects .../Plug-ins/"
    echo ""
fi

# CinematicFX macOS Build Instructions

## Prerequisites

### 1. Install Xcode Command Line Tools
```bash
xcode-select --install
```

### 2. Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. Install CMake
```bash
brew install cmake
```

### 4. Download Adobe After Effects SDK
- Download: Adobe After Effects SDK 25.6_61 for macOS
- Extract to: `~/Downloads/AfterEffectsSDK_25.6_61_mac/`

## Build Instructions

### Option 1: Using Build Script (Recommended)
```bash
# Make script executable
chmod +x build_plugin_mac.sh

# Build plugin
./build_plugin_mac.sh
```

### Option 2: Manual Build
```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCINEMATICFX_BUILD_METAL=ON \
      -DCINEMATICFX_BUILD_CUDA=OFF \
      ..

# Build with all CPU cores
make -j$(sysctl -n hw.ncpu)
```

### Option 3: Using Xcode
```bash
# Generate Xcode project
mkdir -p xcode && cd xcode
cmake -G Xcode \
      -DCINEMATICFX_BUILD_METAL=ON \
      ..

# Open in Xcode
open CinematicFX.xcodeproj

# Build in Xcode: Product → Build (⌘B)
```

## Installation

### Option 1: Using Install Script
```bash
chmod +x install_mac.sh
./install_mac.sh
```

### Option 2: Manual Installation

**For Premiere Pro:**
```bash
sudo cp -R build/CinematicFX.plugin "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
```

**For After Effects:**
```bash
sudo cp -R build/CinematicFX.plugin "/Applications/Adobe After Effects 2025/Plug-ins/"
```

**User-specific (no sudo needed):**
```bash
cp -R build/CinematicFX.plugin "~/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
```

## Usage in Premiere Pro/After Effects

1. Restart Premiere Pro or After Effects
2. Go to: **Effects → Video Effects → CinematicFX**
3. Drag **CinematicFX** effect onto your video clip
4. Adjust parameters in Effect Controls panel

### Default Settings (Optimized for Visibility)
- Bloom Amount: 50, Radius: 40
- Glow Intensity: 80, Threshold: 70
- Halation Intensity: 60, Radius: 15
- Grain Amount: 35, Size: 1.0
- Chromatic Aberration: 0 (disabled by default)

## GPU Acceleration

The macOS version uses **Metal** for GPU acceleration:
- Automatically uses Metal on compatible Macs
- Falls back to CPU on older systems
- No configuration needed - works automatically

### Check GPU Usage
```bash
# View Metal-capable GPUs
system_profiler SPDisplaysDataType | grep Metal

# Monitor GPU usage while rendering
sudo powermetrics --samplers gpu_power
```

## Troubleshooting

### Build Errors

**"Adobe SDK not found"**
```bash
# Verify SDK path
ls ~/Downloads/AfterEffectsSDK_25.6_61_mac/

# Update path in CMakeLists.txt if different
```

**"Metal framework not found"**
```bash
# Check macOS version (need 10.13+)
sw_vers

# Update Xcode Command Line Tools
softwareupdate --all --install --force
```

**"CMake not found"**
```bash
brew install cmake
# Or download from: https://cmake.org/download/
```

### Plugin Not Appearing

**Check installation:**
```bash
# Verify plugin exists
ls -la "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/CinematicFX.plugin"

# Check permissions
ls -la "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
```

**Fix permissions:**
```bash
sudo chmod -R 755 "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/CinematicFX.plugin"
```

**Clear plugin cache:**
```bash
# Premiere Pro
rm -rf ~/Library/Caches/Adobe/Premiere\ Pro/

# After Effects
rm -rf ~/Library/Caches/Adobe/After\ Effects/
```

### Performance Issues

**Slow rendering:**
- Reduce Radius values to 20-30
- Disable effects you're not using
- Check Activity Monitor for CPU/GPU usage

**For 4K footage:**
```bash
# Set quality preset in code or reduce radius:
# Bloom Radius: 40 → 25
# Glow Radius: 40 → 25
# Halation Radius: 15 → 10
```

## Development

### Enable Debug Build
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(sysctl -n hw.ncpu)
```

### View Build Log
```bash
make VERBOSE=1
```

### Code Signing (Optional)
```bash
# Sign the plugin
codesign --force --sign - build/CinematicFX.plugin

# Verify signature
codesign -v build/CinematicFX.plugin
```

## System Requirements

- **OS:** macOS 10.13 High Sierra or later
- **CPU:** Intel or Apple Silicon (M1/M2/M3)
- **RAM:** 8 GB minimum, 16 GB recommended
- **GPU:** Metal-capable (most Macs 2012+)
- **Apps:** Premiere Pro 2024+ or After Effects 2024+

## Build Output

Successful build creates:
- `build/CinematicFX.plugin` - Main plugin bundle
- Plugin size: ~200-300 KB (depends on Metal shaders)
- Universal binary (Intel + Apple Silicon)

## Performance Expectations

### With Metal GPU Acceleration
- HD (1080p): Real-time preview (30-60 fps)
- 4K (2160p): 15-25 fps preview, fast export
- 8K: Preview rendering recommended

### CPU Fallback (older Macs)
- HD: 15-30 fps preview
- 4K: 5-15 fps preview
- Fully functional, just slower

## Next Steps

After successful build and installation:
1. Test all 5 effects in Premiere Pro
2. Verify GPU acceleration is working (check Activity Monitor)
3. Test performance with your footage
4. Create presets for common looks
5. Share feedback or issues

---

**Support:** For issues, check GitHub Issues or contact developer
**Version:** 1.0.0 (Production Release)
**Build Date:** December 18, 2025

# CinematicFX - Build Instructions

Complete guide for building the plugin from source code.

---

## Prerequisites

### All Platforms
- **CMake** 3.20 or later
- **Git** (for version control)
- **Adobe After Effects SDK** (download from Adobe Developer Console)

### Windows
- **Visual Studio 2019 or later** (with C++ workload)
- **CUDA Toolkit 12.0+** (for NVIDIA GPU support)
  - Download from: https://developer.nvidia.com/cuda-downloads
- **Windows 10 SDK** (included with Visual Studio)

### macOS
- **Xcode 14 or later** (with Command Line Tools)
- **Metal SDK** (included with Xcode)
- **macOS 10.14+** (for Metal support)

---

## Setup Adobe SDK

1. **Download Adobe AE SDK:**
   - Visit: https://developer.adobe.com/console/
   - Navigate to After Effects SDK
   - Download latest version (24.0+)

2. **Extract SDK:**
   ```bash
   # Windows
   Expand-Archive -Path AfterEffectsSDK.zip -DestinationPath C:\SDKs\AdobeAE
   
   # macOS/Linux
   unzip AfterEffectsSDK.zip -d ~/SDKs/AdobeAE
   ```

3. **Set Environment Variable:**
   ```bash
   # Windows (PowerShell)
   [Environment]::SetEnvironmentVariable("ADOBE_SDK_ROOT", "C:\SDKs\AdobeAE", "User")
   
   # macOS/Linux
   export ADOBE_SDK_ROOT=~/SDKs/AdobeAE
   ```

---

## Build Instructions

### Windows (CUDA + CPU)

```powershell
# 1. Clone repository
git clone https://github.com/polcasals/CinematicFX.git
cd CinematicFX

# 2. Create build directory
mkdir build
cd build

# 3. Configure with CMake
cmake .. -G "Visual Studio 17 2022" `
    -DADOBE_SDK_ROOT="C:\SDKs\AdobeAE" `
    -DCINEMATICFX_BUILD_CUDA=ON `
    -DCINEMATICFX_BUILD_TESTS=ON

# 4. Build (Release configuration)
cmake --build . --config Release

# Output: build/Release/CinematicFX.prm
```

**Build Options:**
- `-DCINEMATICFX_BUILD_CUDA=OFF` - Disable CUDA support
- `-DCINEMATICFX_BUILD_TESTS=OFF` - Skip unit tests
- `-DCMAKE_BUILD_TYPE=Debug` - Build debug version

---

### macOS (Metal + CPU)

```bash
# 1. Clone repository
git clone https://github.com/polcasals/CinematicFX.git
cd CinematicFX

# 2. Create build directory
mkdir build && cd build

# 3. Configure with CMake
cmake .. \
    -DADOBE_SDK_ROOT="$HOME/SDKs/AdobeAE" \
    -DCINEMATICFX_BUILD_METAL=ON \
    -DCINEMATICFX_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release

# 4. Build
cmake --build . --config Release

# Output: build/CinematicFX.plugin
```

**Universal Binary (Intel + Apple Silicon):**
```bash
cmake .. \
    -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DADOBE_SDK_ROOT="$HOME/SDKs/AdobeAE"
```

---

## Testing

### Run Unit Tests

```bash
# Windows
.\build\Release\CinematicFXTests.exe

# macOS/Linux
./build/CinematicFXTests
```

**Expected Output:**
```
[==========] Running 24 tests from 6 test suites.
[----------] Global test environment set-up.
[----------] 4 tests from BloomEffect
[ RUN      ] BloomEffect.ParameterValidation
[       OK ] BloomEffect.ParameterValidation (1 ms)
[ RUN      ] BloomEffect.ZeroIntensitySkip
[       OK ] BloomEffect.ZeroIntensitySkip (0 ms)
...
[==========] 24 tests from 6 test suites ran. (2341 ms total)
[  PASSED  ] 24 tests.
```

### Performance Benchmark

```bash
# Windows
.\build\Release\CinematicFXTests.exe --benchmark

# macOS
./build/CinematicFXTests --benchmark
```

---

## Installation

### Windows

```powershell
# Manual installation
Copy-Item build\Release\CinematicFX.prm `
    "C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\"

# Or use installer
cmake --build . --target install
```

### macOS

```bash
# Manual installation
sudo cp -R build/CinematicFX.plugin \
    "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"

# Or use installer
sudo cmake --build . --target install
```

### Verify Installation

1. Launch Adobe Premiere Pro
2. Go to **Window** → **Effects**
3. Search for "CinematicFX"
4. Should appear under **Video Effects** → **CinematicFX**

---

## Troubleshooting

### "Adobe SDK not found"

**Solution:**
```bash
# Set ADOBE_SDK_ROOT when running CMake
cmake .. -DADOBE_SDK_ROOT="/path/to/adobe/sdk"
```

### "CUDA not found" (Windows)

**Solution:**
1. Install CUDA Toolkit from NVIDIA
2. Add to PATH:
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
   ```
3. Rerun CMake

### "Metal framework not found" (macOS)

**Solution:**
- Install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Build errors in Visual Studio

**Solution:**
- Ensure C++ workload installed
- Use "Developer Command Prompt for VS 2022"
- Try cleaning build:
  ```powershell
  cmake --build . --target clean
  cmake --build . --config Release
  ```

### Plugin not loading in Premiere Pro

**Check:**
1. Plugin installed to correct path
2. Premiere Pro version 24.0+
3. Check Windows Event Viewer / macOS Console for errors
4. Ensure dependencies (CUDA runtime, etc.) installed

---

## Advanced Build Options

### Debug Build with Symbols

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCINEMATICFX_VERBOSE_LOGGING=ON
```

### Cross-Compilation (macOS for both Intel/ARM)

```bash
cmake .. -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
```

### Specific CUDA Architectures

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
```

### Static Linking (reduce dependencies)

```bash
cmake .. -DCINEMATICFX_STATIC_RUNTIME=ON
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build CinematicFX

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install CUDA
        uses: Jimver/cuda-toolkit@v0.2.11
      - name: Configure
        run: cmake -B build -DADOBE_SDK_ROOT=${{ secrets.ADOBE_SDK_PATH }}
      - name: Build
        run: cmake --build build --config Release
      - name: Test
        run: .\build\Release\CinematicFXTests.exe

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure
        run: cmake -B build -DADOBE_SDK_ROOT=${{ secrets.ADOBE_SDK_PATH }}
      - name: Build
        run: cmake --build build --config Release
      - name: Test
        run: ./build/CinematicFXTests
```

---

## Code Signing (Distribution)

### Windows

```powershell
# Sign .prm with Authenticode
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com `
    build\Release\CinematicFX.prm
```

### macOS

```bash
# Sign plugin bundle
codesign --force --sign "Developer ID Application: Your Name" \
    --timestamp build/CinematicFX.plugin

# Notarize for Gatekeeper
xcrun notarytool submit build/CinematicFX.plugin.zip \
    --apple-id your@email.com --password app-specific-password --wait
```

---

## Creating Installers

### Windows (NSIS)

```nsis
# installer.nsi
!include "MUI2.nsh"

Name "CinematicFX"
OutFile "CinematicFX_Installer.exe"
InstallDir "C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore"

Section "Install"
    SetOutPath $INSTDIR
    File "build\Release\CinematicFX.prm"
SectionEnd
```

Build:
```powershell
makensis installer.nsi
```

### macOS (.dmg)

```bash
# Create disk image
hdiutil create -volname "CinematicFX" -srcfolder build/CinematicFX.plugin \
    -ov -format UDZO CinematicFX_Installer.dmg
```

---

**Document Version:** 1.0.0  
**Last Updated:** December 2025

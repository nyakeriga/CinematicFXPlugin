# Build Instructions - CinematicFX Plugin

## Current Status

✅ **CUDA Backend** - Complete implementation  
✅ **CPU Fallback** - Complete implementation  
✅ **Core Systems** - Logger, Math Utils, Render Pipeline  
⏳ **Adobe SDK Integration** - Requires SDK download  

## Issues Fixed

1. ✅ CUDA include paths configured in VS Code
2. ✅ Conditional compilation for CUDA availability
3. ✅ Standalone build system (no Adobe SDK required for testing)

## Quick Start - Test CUDA Installation

### Option 1: Simple CUDA Test
```powershell
.\test_cuda.ps1
```

This will compile and run a simple CUDA test program to verify your installation.

### Option 2: Build Core Components
```powershell
.\build_standalone.ps1
```

This builds the core plugin components (GPU backends, effects) without Adobe SDK integration.

## Full Build - With Adobe SDK

### Prerequisites

1. **CUDA Toolkit 12.0** ✅ (Installing/Installed)
2. **Adobe After Effects SDK** ❌ (Required - Download from Adobe)
3. **Visual Studio 2022** (with C++ tools)

### Download Adobe After Effects SDK

1. Visit: https://developer.adobe.com/after-effects/
2. Download: After Effects SDK (latest version)
3. Extract to: `C:\Adobe\AfterEffectsSDK` (or your preferred location)

### Configure Build

Edit `CMakeLists.txt` and set:
```cmake
set(AE_SDK_PATH "C:/Adobe/AfterEffectsSDK" CACHE PATH "Adobe After Effects SDK path")
```

### Build Plugin

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Current Build Errors Explained

### Error: Cannot open Adobe SDK headers
```
cannot open source file "AEConfig.h"
cannot open source file "entry.h"
...
```

**Cause**: Adobe After Effects SDK not installed  
**Solution**: Download and install Adobe AE SDK (see above)  
**Workaround**: Use standalone build to test CUDA components

### Error: cuda_runtime.h not found
```
cannot open source file "cuda_runtime.h"
```

**Cause**: CUDA Toolkit not installed or not in include path  
**Status**: Should be fixed after CUDA installation completes  
**Solution**: 
1. Wait for CUDA installation to complete
2. Restart VS Code
3. Verify with: `nvcc --version`

## Project Structure

```
CinematicFXPlugin/
├── include/              # Public headers
│   ├── CinematicFX.h
│   ├── EffectParameters.h
│   └── GPUInterface.h
├── src/
│   ├── core/            # Plugin core
│   │   ├── PluginMain.cpp    (requires Adobe SDK)
│   │   └── RenderPipeline.cpp
│   ├── gpu/             # GPU backends
│   │   ├── CUDABackend.cpp   ✅
│   │   ├── CPUFallback.cpp   ✅
│   │   └── TextureManager.cpp
│   ├── kernels/cuda/    # CUDA kernels
│   │   ├── bloom_kernel.cu   ✅
│   │   ├── glow_kernel.cu    ✅
│   │   ├── halation_kernel.cu ✅
│   │   ├── grain_kernel.cu    ✅
│   │   └── chromatic_aberration_kernel.cu ✅
│   └── utils/           # Utilities
│       ├── Logger.cpp        ✅
│       └── MathUtils.cpp     ✅
├── test_cuda.cpp        # Simple CUDA test
└── CMakeLists_Standalone.txt # Build without Adobe SDK
```

## Next Steps

### Immediate (Testing CUDA)
1. ✅ Wait for CUDA installation to complete
2. Run `.\test_cuda.ps1` to verify CUDA works
3. Run `.\build_standalone.ps1` to build core components

### Full Integration (Adobe Plugin)
1. Download Adobe After Effects SDK
2. Update `CMakeLists.txt` with SDK path
3. Build full plugin with `cmake`
4. Test in After Effects / Premiere Pro

## Troubleshooting

### CUDA Installation Issues
- Installation log: Check terminal output
- Verify: `nvcc --version`
- Path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`

### Build Issues
- Restart VS Code after CUDA installation
- Check C++ IntelliSense configuration (`.vscode/c_cpp_properties.json`)
- Use standalone build for testing without Adobe SDK

### Adobe SDK Issues
- Ensure SDK is downloaded from official Adobe Developer site
- Check SDK path in CMakeLists.txt
- SDK version must be compatible with After Effects 2024+

## Support

For issues:
1. Check CUDA installation: `nvcc --version`
2. Verify GPU: `nvidia-smi`
3. Test core components: `.\build_standalone.ps1`
4. Review build errors in VS Code PROBLEMS panel

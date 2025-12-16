# Project Structure - File Tree

```
CinematicFXPlugin/
│
├── 📄 README.md                           # Project overview & quick start
├── 📄 LICENSE                             # Commercial license terms
├── 📄 CMakeLists.txt                      # Cross-platform build system
├── 📄 ARCHITECTURE.md                     # Detailed architecture document
├── 📄 .gitignore                          # Git ignore rules
│
├── 📁 include/                            # Public API headers
│   ├── CinematicFX.h                     # Main plugin header
│   ├── EffectParameters.h                # Parameter structures & presets
│   └── GPUInterface.h                    # GPU abstraction interface
│
├── 📁 src/                                # Source code implementation
│   │
│   ├── 📁 core/                          # Core plugin infrastructure
│   │   ├── PluginMain.cpp                # Adobe AE SDK entry point (REQUIRED)
│   │   ├── ParameterManager.cpp          # Keyframe parameter handling
│   │   ├── ParameterManager.h
│   │   ├── RenderPipeline.cpp            # Master rendering coordinator
│   │   ├── RenderPipeline.h
│   │   ├── ColorManagement.cpp           # 32-bit float color space handling
│   │   ├── ColorManagement.h
│   │   ├── LicenseManager.cpp            # License validation & activation
│   │   └── LicenseManager.h
│   │
│   ├── 📁 gpu/                           # GPU Abstraction Layer
│   │   ├── GPUBackend.h                  # Abstract GPU backend base class
│   │   ├── GPUContext.cpp                # GPU initialization & management
│   │   │
│   │   ├── CUDABackend.h                 # NVIDIA CUDA backend interface
│   │   ├── CUDABackend.cpp               # CUDA implementation
│   │   │
│   │   ├── MetalBackend.h                # Apple Metal backend interface
│   │   ├── MetalBackend.mm               # Metal implementation (Obj-C++)
│   │   │
│   │   ├── CPUFallback.h                 # Software fallback interface
│   │   ├── CPUFallback.cpp               # CPU SIMD implementation
│   │   │
│   │   ├── TextureManager.cpp            # GPU texture pool management
│   │   └── TextureManager.h
│   │
│   ├── 📁 effects/                       # Effect Implementation Modules
│   │   ├── EffectBase.h                  # Abstract effect interface
│   │   ├── EffectBase.cpp
│   │   │
│   │   ├── BloomEffect.h                 # Atmospheric bloom effect
│   │   ├── BloomEffect.cpp
│   │   │
│   │   ├── GlowEffect.h                  # Highlight diffusion (Pro-Mist)
│   │   ├── GlowEffect.cpp
│   │   │
│   │   ├── HalationEffect.h              # Red film fringe effect
│   │   ├── HalationEffect.cpp
│   │   │
│   │   ├── GrainEffect.h                 # Curated cinematic grain
│   │   ├── GrainEffect.cpp
│   │   │
│   │   ├── ChromaticAberration.h         # Color channel shift effect
│   │   └── ChromaticAberration.cpp
│   │
│   ├── 📁 kernels/                       # GPU Compute Kernels
│   │   │
│   │   ├── 📁 cuda/                      # CUDA kernels (.cu files)
│   │   │   ├── bloom_kernel.cu           # Bloom GPU kernel
│   │   │   ├── glow_kernel.cu            # Glow GPU kernel
│   │   │   ├── halation_kernel.cu        # Halation GPU kernel
│   │   │   ├── grain_kernel.cu           # Grain GPU kernel
│   │   │   ├── chromatic_aberration_kernel.cu
│   │   │   └── common_utils.cuh          # Shared CUDA utilities
│   │   │
│   │   └── 📁 metal/                     # Metal shaders (.metal files)
│   │       ├── bloom_shader.metal        # Bloom Metal shader
│   │       ├── glow_shader.metal         # Glow Metal shader
│   │       ├── halation_shader.metal     # Halation Metal shader
│   │       ├── grain_shader.metal        # Grain Metal shader
│   │       ├── chromatic_aberration_shader.metal
│   │       └── common_utils.metal        # Shared Metal utilities
│   │
│   ├── 📁 utils/                         # Utility & Helper Functions
│   │   ├── MathUtils.h                   # Vector math, interpolation
│   │   ├── MathUtils.cpp
│   │   ├── Logger.h                      # Debug & performance logging
│   │   ├── Logger.cpp
│   │   ├── PerformanceTimer.h            # GPU/CPU profiling
│   │   ├── PerformanceTimer.cpp
│   │   ├── ErrorHandler.h                # Exception & error recovery
│   │   └── ErrorHandler.cpp
│   │
│   └── 📁 ui/                            # User Interface Layer
│       ├── ParameterDefinitions.cpp      # UI parameter definitions
│       ├── ParameterDefinitions.h
│       ├── PresetManager.cpp             # User preset save/load
│       └── PresetManager.h
│
├── 📁 resources/                         # Assets & Resources
│   ├── PiPL.r                            # Plugin Property List (macOS)
│   ├── PiPL.rc                           # Plugin Resource (Windows)
│   ├── Info.plist                        # macOS bundle info
│   │
│   ├── 📁 icons/                         # UI icons
│   │   ├── plugin_icon.png
│   │   └── effect_icons/
│   │       ├── bloom.png
│   │       ├── glow.png
│   │       ├── halation.png
│   │       ├── grain.png
│   │       └── chromatic_aberration.png
│   │
│   └── 📁 presets/                       # Factory presets
│       ├── cinematic_glow.preset
│       ├── vintage_film.preset
│       ├── dreamy_diffusion.preset
│       └── subtle_grain.preset
│
├── 📁 tests/                             # Unit & Integration Tests
│   ├── test_bloom.cpp                    # Bloom effect tests
│   ├── test_glow.cpp                     # Glow effect tests
│   ├── test_halation.cpp                 # Halation effect tests
│   ├── test_grain.cpp                    # Grain effect tests
│   ├── test_chromatic_aberration.cpp     # Chromatic aberration tests
│   ├── test_gpu_fallback.cpp             # GPU fallback mechanism tests
│   ├── test_parameter_validation.cpp     # Parameter clamping tests
│   ├── benchmark_suite.cpp               # Performance benchmarks
│   └── test_main.cpp                     # Test runner main
│
├── 📁 docs/                              # Documentation
│   ├── TECHNICAL_SPEC.md                 # Technical specification
│   ├── USER_GUIDE.md                     # User manual
│   ├── API_REFERENCE.md                  # Developer API docs
│   ├── BUILD.md                          # Build instructions
│   ├── PERFORMANCE.md                    # Performance optimization guide
│   └── CHANGELOG.md                      # Version history
│
├── 📁 build/                             # Build output (gitignored)
│   ├── Release/
│   │   ├── CinematicFX.prm              # Windows plugin
│   │   └── CinematicFX.plugin/          # macOS plugin bundle
│   └── Debug/
│
├── 📁 installers/                        # Installer scripts
│   ├── windows/
│   │   ├── installer.nsi                # NSIS installer script
│   │   └── setup.iss                    # Inno Setup script
│   └── macos/
│       ├── create_dmg.sh                # DMG creation script
│       └── postinstall.sh               # Post-install script
│
└── 📁 ci/                                # Continuous Integration
    ├── .github/
    │   └── workflows/
    │       ├── build_windows.yml
    │       ├── build_macos.yml
    │       └── run_tests.yml
    └── scripts/
        ├── setup_build_env.sh
        └── run_benchmarks.sh
```

---

## Key File Descriptions

### Core Plugin Files (CRITICAL)
- **`src/core/PluginMain.cpp`** - Adobe SDK entry point, REQUIRED for plugin to load
- **`resources/PiPL.r` / `PiPL.rc`** - Plugin metadata, defines plugin name, category, version

### GPU Abstraction (ARCHITECTURE)
- **`src/gpu/GPUContext.cpp`** - Automatic backend selection (CUDA → Metal → CPU)
- **`src/gpu/CUDABackend.cpp`** - NVIDIA GPU implementation
- **`src/gpu/MetalBackend.mm`** - Apple GPU implementation (Objective-C++)
- **`src/gpu/CPUFallback.cpp`** - Software fallback (SIMD optimized)

### Effect Implementations
- Each effect has `.h` header + `.cpp` implementation
- GPU kernels in `src/kernels/cuda/*.cu` and `src/kernels/metal/*.metal`

### Build System
- **`CMakeLists.txt`** - Cross-platform build configuration
- Automatically detects CUDA/Metal availability
- Builds `.prm` (Windows) or `.plugin` (macOS)

### Documentation
- **`ARCHITECTURE.md`** - High-level architecture design
- **`docs/TECHNICAL_SPEC.md`** - Detailed algorithms & specifications
- **`docs/BUILD.md`** - Compilation instructions

---

## Build Output Locations

**Windows:**
```
build/Release/CinematicFX.prm
```

**macOS:**
```
build/CinematicFX.plugin/
  └── Contents/
      ├── Info.plist
      ├── MacOS/
      │   └── CinematicFX (binary)
      └── Resources/
          └── CinematicFX.metallib (Metal shaders)
```

---

## Next Steps for Implementation

### Phase 1: Foundation (Week 1-2)
1. Implement `PluginMain.cpp` (Adobe SDK integration)
2. Implement `ParameterManager.cpp` (keyframe handling)
3. Implement `GPUContext.cpp` (backend detection)
4. Test: Plugin loads in Premiere Pro

### Phase 2: GPU Backends (Week 3)
1. Implement `CUDABackend.cpp` + basic CUDA kernel
2. Implement `MetalBackend.mm` + basic Metal shader
3. Implement `CPUFallback.cpp` + SIMD blur
4. Test: All backends initialize correctly

### Phase 3: Effects (Week 4-5)
1. Implement each effect (Bloom, Glow, Halation, Grain, Chromatic Aberration)
2. Implement corresponding GPU kernels (CUDA + Metal + CPU)
3. Test: Each effect produces correct output

### Phase 4: Polish (Week 6)
1. Performance optimization (profiling, memory management)
2. License system integration
3. Documentation finalization
4. Installer creation

---

**File Count Summary:**
- Header files: ~25
- Source files: ~30
- GPU kernels: ~10
- Documentation: ~8
- Total: ~73 files

**Lines of Code Estimate:**
- C++ code: ~15,000 lines
- CUDA kernels: ~2,000 lines
- Metal shaders: ~2,000 lines
- Total: ~19,000 lines

---

**Document Version:** 1.0.0  
**Status:** Production Architecture

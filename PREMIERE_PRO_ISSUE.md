# CinematicFX Plugin - Important Installation Notes

## ⚠️ CURRENT LIMITATION: Adobe After Effects Only

### The Issue
Premiere Pro logs show: **"No loaders recognized this plugin"**

This means Premiere Pro cannot recognize the plugin format. The plugin currently works with:
- ✅ **Adobe After Effects** (primary target)
- ❌ **Adobe Premiere Pro** (requires embedded PiPL resource)

### Why This Happens
The plugin is missing an embedded **PiPL (Plugin Property List) resource** that Premiere Pro requires to recognize .prm plugins. After Effects is more lenient and can load plugins based on exported entry points alone.

### Solution Options

#### Option 1: Use in After Effects (WORKS NOW)
1. Copy both files to After Effects plugins folder:
   ```
   C:\Program Files\Adobe\Adobe After Effects [VERSION]\Support Files\Plug-ins\
   ```
   - CinematicFX.prm
   - vcruntime140.dll

2. Restart After Effects

3. Find effect at: **Effect → CinematicFX → CinematicFX**

#### Option 2: Fix for Premiere Pro (Requires Development)
To make this work in Premiere Pro, we need to:

1. **Create proper PiPL resource file**
   - Windows .rc files don't support Mac PiPL syntax
   - Need to use Adobe's PiPL compiler tool (PiPLTool.exe) from SDK
   - Or create binary PiPL resource manually

2. **Use Adobe's PiPL Compiler**
   ```
   # Location in SDK (if available):
   AfterEffectsSDK/Examples/Resources/PiPLTool.exe
   
   # Compile PiPL.r to PiPL.rc:
   PiPLTool.exe PiPL.r PiPL.rc
   ```

3. **Embed in plugin**
   - Add compiled PiPL.rc to CMakeLists.txt
   - Rebuild plugin
   - Test in Premiere Pro

### Current Working Configuration

**Plugin File:** `C:\Users\Admin\CinematicFXPlugin\build\Release\CinematicFX.prm`

**Installation for After Effects:**
```powershell
# Copy plugin + runtime DLL
Copy-Item "C:\Users\Admin\CinematicFXPlugin\build\Release\CinematicFX.prm" `
          "C:\Program Files\Adobe\Adobe After Effects 2024\Support Files\Plug-ins\"
          
Copy-Item "C:\Users\Admin\CinematicFXPlugin\installer\vcruntime140.dll" `
          "C:\Program Files\Adobe\Adobe After Effects 2024\Support Files\Plug-ins\"
```

### Alternative: Install Visual C++ Redistributable
Instead of copying vcruntime140.dll:
1. Install: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Then only copy CinematicFX.prm

### Effect Features (Working in After Effects)

All 5 professional cinematic effects:

1. **Bloom**
   - Amount (0-100)
   - Radius (1-100)
   - Tint Color

2. **Glow (Pro-Mist)**
   - Threshold (0-100)
   - Radius (1-100)
   - Intensity (0-200)

3. **Halation (Film Fringe)**
   - Intensity (0-100)
   - Radius (1-50)

4. **Curated Grain**
   - Amount (0-100)
   - Size (0.5-5.0)
   - Luma Mapping (0-100)

5. **Chromatic Aberration**
   - Amount (0-10)
   - Angle (0-360°)

### Technical Details

- **Version:** 1.0.0
- **Architecture:** x64
- **SDK:** Adobe After Effects SDK 25.6
- **Runtime:** CPU Fallback (universal - works on all machines)
- **Color Depth:** 32-bit float HDR pipeline
- **Dependencies:** vcruntime140.dll (Visual C++ Runtime)

### Next Steps to Support Premiere Pro

1. **Locate Adobe PiPL Tool** in SDK downloads
2. **Compile PiPL resource** from existing PiPL.r file
3. **Embed in build** and test in Premiere Pro
4. **Alternative:** Study existing Premiere-compatible .prm plugins to understand binary PiPL format

### For Now: Use After Effects
The plugin is fully functional in **Adobe After Effects** with all 5 effects working perfectly. Premiere Pro support requires additional PiPL resource work.

---

**Built for:** Pol Casals  
**Date:** December 2025  
**Status:** Production-ready for After Effects, Premiere Pro requires PiPL resource

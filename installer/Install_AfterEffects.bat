@echo off
REM CinematicFX Plugin Installer for Adobe After Effects
REM This script copies the plugin files to the Adobe After Effects plugins directory

echo ================================================================================
echo CinematicFX Plugin Installer - After Effects
echo ================================================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This installer requires Administrator privileges.
    echo Please right-click and select "Run as Administrator"
    echo.
    pause
    exit /b 1
)

REM Define source directory
set "SOURCE_DIR=%~dp0"

REM Try to find After Effects installation
set "AE_DIR="
for %%v in (2025 2024 2023 2022) do (
    if exist "C:\Program Files\Adobe\Adobe After Effects %%v\Support Files\Plug-ins\" (
        set "AE_DIR=C:\Program Files\Adobe\Adobe After Effects %%v\Support Files\Plug-ins\"
        set "AE_VERSION=%%v"
        goto :found
    )
)

:found
if "%AE_DIR%"=="" (
    echo ERROR: Adobe After Effects not found.
    echo.
    echo Please install the plugin manually:
    echo 1. Find your After Effects plugins folder
    echo 2. Copy CinematicFX.prm and vcruntime140.dll there
    echo 3. Restart After Effects
    echo.
    pause
    exit /b 1
)

echo Found After Effects %AE_VERSION%
echo Destination: %AE_DIR%
echo.

echo Copying plugin files...
echo.

REM Copy plugin file
if exist "%SOURCE_DIR%CinematicFX.prm" (
    copy /Y "%SOURCE_DIR%CinematicFX.prm" "%AE_DIR%CinematicFX.prm"
    if %errorLevel% equ 0 (
        echo [OK] CinematicFX.prm copied successfully
    ) else (
        echo [FAILED] Could not copy CinematicFX.prm
        pause
        exit /b 1
    )
) else (
    echo [ERROR] CinematicFX.prm not found in installer folder!
    pause
    exit /b 1
)

REM Copy runtime DLL
if exist "%SOURCE_DIR%vcruntime140.dll" (
    copy /Y "%SOURCE_DIR%vcruntime140.dll" "%AE_DIR%vcruntime140.dll"
    if %errorLevel% equ 0 (
        echo [OK] vcruntime140.dll copied successfully
    ) else (
        echo [FAILED] Could not copy vcruntime140.dll
        pause
        exit /b 1
    )
) else (
    echo [WARNING] vcruntime140.dll not found
    echo The plugin may not work without the Visual C++ runtime
    echo Please install: https://aka.ms/vs/17/release/vc_redist.x64.exe
)

echo.
echo ================================================================================
echo INSTALLATION COMPLETE!
echo ================================================================================
echo.
echo Files installed to: %AE_DIR%
echo.
echo NEXT STEPS:
echo   1. Restart Adobe After Effects %AE_VERSION% (if it's running)
echo   2. Open or create a composition
echo   3. Find the effect at: Effect -^> CinematicFX -^> CinematicFX
echo.
echo AVAILABLE EFFECTS:
echo   - Bloom (Amount, Radius, Tint)
echo   - Glow / Pro-Mist (Threshold, Radius, Intensity)
echo   - Halation / Film Fringe (Intensity, Radius)
echo   - Curated Grain (Amount, Size, Luma Mapping)
echo   - Chromatic Aberration (Amount, Angle)
echo.
echo NOTE: This plugin currently works in After Effects only.
echo       Premiere Pro support requires additional PiPL resource development.
echo.
pause

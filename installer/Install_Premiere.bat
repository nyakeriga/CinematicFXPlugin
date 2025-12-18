@echo off
REM CinematicFX Plugin Installer for Adobe Premiere Pro
REM This script copies the plugin files to the Adobe Common plugins directory

echo ================================================================================
echo CinematicFX Plugin Installer
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

REM Define source and destination paths
set "SOURCE_DIR=%~dp0"
set "DEST_DIR=C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore"

echo Source folder: %SOURCE_DIR%
echo Destination: %DEST_DIR%
echo.

REM Check if destination exists
if not exist "%DEST_DIR%" (
    echo WARNING: Adobe Common Plugins folder not found.
    echo Creating directory...
    mkdir "%DEST_DIR%"
    if %errorLevel% neq 0 (
        echo ERROR: Failed to create directory. Adobe Premiere Pro may not be installed.
        echo.
        echo Please manually copy these files to your Adobe plugins folder:
        echo   - CinematicFX.prm
        echo   - vcruntime140.dll
        echo.
        pause
        exit /b 1
    )
)

echo Copying plugin files...
echo.

REM Copy plugin file
if exist "%SOURCE_DIR%CinematicFX.prm" (
    copy /Y "%SOURCE_DIR%CinematicFX.prm" "%DEST_DIR%\CinematicFX.prm"
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
    copy /Y "%SOURCE_DIR%vcruntime140.dll" "%DEST_DIR%\vcruntime140.dll"
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
echo Files installed to: %DEST_DIR%
echo.
echo NEXT STEPS:
echo   1. Restart Adobe Premiere Pro (if it's running)
echo   2. Open or create a project
echo   3. Find the effect at: Effects -^> Video Effects -^> CinematicFX -^> CinematicFX
echo.
echo If the plugin doesn't appear:
echo   - Make sure Premiere Pro is completely closed and restarted
echo   - Check Windows Security hasn't blocked the files
echo   - Try installing Visual C++ Redistributable (see instructions)
echo.
pause

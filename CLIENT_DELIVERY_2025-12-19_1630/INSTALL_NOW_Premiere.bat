@echo off
echo ========================================
echo CinematicFX Installer (Premiere Pro)
echo CRITICAL FIXES BUILD - December 18, 2025
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Not running as Administrator
    echo Some installations may fail without admin rights
    echo.
    timeout /t 3 >nul
)

REM Premiere Pro plugin paths (check multiple versions)
set "PREMIERE_2025=C:\Program Files\Adobe\Adobe Premiere Pro 2025\Plug-ins\Common"
set "PREMIERE_2024=C:\Program Files\Adobe\Adobe Premiere Pro 2024\Plug-ins\Common"
set "PREMIERE_MEDIACORE=C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore"

echo Installing CinematicFX plugin for Premiere Pro...
echo Plugin version: 1.0.0 (53 KB with crash fixes)
echo.

REM Check if Premiere is running
tasklist /FI "IMAGENAME eq Adobe Premiere Pro.exe" 2>NUL | find /I /N "Adobe Premiere Pro.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo WARNING: Premiere Pro is currently running!
    echo Please close Premiere Pro before installing.
    echo.
    pause
    exit /b
)

set INSTALLED=0

REM Try Premiere 2025
if exist "%PREMIERE_2025%" (
    echo [1/3] Installing to Premiere Pro 2025...
    copy /Y "CinematicFX.prm" "%PREMIERE_2025%\" >nul 2>&1
    copy /Y "vcruntime140.dll" "%PREMIERE_2025%\" >nul 2>&1
    if exist "%PREMIERE_2025%\CinematicFX.prm" (
        echo      [OK] Installed to: %PREMIERE_2025%
        set INSTALLED=1
    )
)

REM Try Premiere 2024
if exist "%PREMIERE_2024%" (
    echo [2/3] Installing to Premiere Pro 2024...
    copy /Y "CinematicFX.prm" "%PREMIERE_2024%\" >nul 2>&1
    copy /Y "vcruntime140.dll" "%PREMIERE_2024%\" >nul 2>&1
    if exist "%PREMIERE_2024%\CinematicFX.prm" (
        echo      [OK] Installed to: %PREMIERE_2024%
        set INSTALLED=1
    )
)

REM Try MediaCore (shared location)
if exist "%PREMIERE_MEDIACORE%" (
    echo [3/3] Installing to shared plugin folder...
    copy /Y "CinematicFX.prm" "%PREMIERE_MEDIACORE%\" >nul 2>&1
    copy /Y "vcruntime140.dll" "%PREMIERE_MEDIACORE%\" >nul 2>&1
    if exist "%PREMIERE_MEDIACORE%\CinematicFX.prm" (
        echo      [OK] Installed to: %PREMIERE_MEDIACORE%
        set INSTALLED=1
    )
)

echo.
if %INSTALLED%==1 (
    echo ========================================
    echo  Installation Successful!
    echo ========================================
    echo.
    echo CRITICAL FIXES APPLIED:
    echo  [OK] Fixed GPU initialization crashes
    echo  [OK] Fixed buffer stride calculation errors
    echo  [OK] Added comprehensive error handling
    echo  [OK] Increased effect visibility (defaults improved)
    echo  [OK] CPU backend for universal compatibility
    echo.
    echo Next steps:
    echo  1. Start Premiere Pro
    echo  2. Go to: Effects - Video Effects - CinematicFX
    echo  3. Drag "CinematicFX" effect onto your video clip
    echo  4. Adjust parameters in Effect Controls panel
    echo.
    echo Default effects (enabled at moderate strength):
    echo  * Bloom (Amount: 50, Radius: 40)
    echo  * Glow (Intensity: 80, Threshold: 70)
    echo  * Halation (Intensity: 60, Radius: 15)
    echo  * Grain (Amount: 35, Size: 1.0)
    echo  * Chromatic Aberration (disabled by default)
    echo.
    echo For troubleshooting, see CRITICAL_FIXES.txt
    echo.
) else (
    echo ========================================
    echo  Automatic Installation Failed
    echo ========================================
    echo.
    echo Could not find Premiere Pro plugin folders.
    echo.
    echo MANUAL INSTALLATION:
    echo  1. Locate your Premiere Pro installation folder
    echo  2. Navigate to: Plug-ins\Common\
    echo  3. Copy these 2 files there:
    echo     * CinematicFX.prm (53 KB)
    echo     * vcruntime140.dll (111 KB)
    echo  4. Restart Premiere Pro
    echo.
    echo Common Premiere plugin locations:
    echo  * C:\Program Files\Adobe\Adobe Premiere Pro 2025\Plug-ins\Common\
    echo  * C:\Program Files\Adobe\Adobe Premiere Pro 2024\Plug-ins\Common\
    echo  * C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\
    echo.
)

pause

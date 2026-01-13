# Build CinematicFX Plugin with Adobe SDK
# Full production build

Write-Host "Building CinematicFX Plugin..." -ForegroundColor Cyan

# Verify Adobe SDK exists
$sdkPath = "C:\Users\Admin\Downloads\AfterEffectsSDK_25.6_61_win\AfterEffectsSDK_25.6_61_win\ae25.6_61.64bit.AfterEffectsSDK\Examples"

if (!(Test-Path $sdkPath)) {
    Write-Host "ERROR: Adobe SDK not found" -ForegroundColor Red
    exit 1
}

Write-Host "SDK found" -ForegroundColor Green

# Create build directory
if (Test-Path "build") {
    Remove-Item -Recurse -Force build
}

New-Item -ItemType Directory -Path "build" | Out-Null
Set-Location build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DADOBE_SDK_ROOT="$sdkPath"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake failed" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Build
cmake --build . --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host "Build successful" -ForegroundColor Green

Set-Location ..

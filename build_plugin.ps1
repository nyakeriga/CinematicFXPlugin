# Build CinematicFX Plugin
# Production build with Adobe SDK and CUDA support

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CinematicFX Plugin Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify Adobe SDK
$sdkPath = "C:\Users\Admin\Downloads\AfterEffectsSDK_25.6_61_win\AfterEffectsSDK_25.6_61_win\ae25.6_61.64bit.AfterEffectsSDK\Examples"

if (-not (Test-Path $sdkPath)) {
    Write-Host "ERROR: Adobe SDK not found" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Adobe SDK found" -ForegroundColor Green

# Check CUDA
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"
if (Test-Path $cudaPath) {
    Write-Host "[OK] CUDA 12.0 found" -ForegroundColor Green
} else {
    Write-Host "[WARN] CUDA not found - CPU fallback only" -ForegroundColor Yellow
}

Write-Host ""

# Create build directory
if (Test-Path "build") {
    Write-Host "Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
}

New-Item -ItemType Directory -Path "build" | Out-Null
Set-Location build

Write-Host "Running CMake configuration..." -ForegroundColor Cyan

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DADOBE_SDK_ROOT="$sdkPath"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "Building plugin..." -ForegroundColor Cyan

cmake --build . --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Build failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Find the plugin
$plugin = Get-ChildItem -Path . -Filter "*.prm" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1

if ($plugin) {
    Write-Host "Plugin: $($plugin.FullName)" -ForegroundColor Cyan
} else {
    Write-Host "DLL files:" -ForegroundColor Cyan
    Get-ChildItem -Path Release -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor White
    }
}

Write-Host ""
Set-Location ..

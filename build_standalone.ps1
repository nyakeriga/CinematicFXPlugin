# Build CinematicFX Plugin (Standalone - No Adobe SDK)
# This builds the core components for testing

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CinematicFX Standalone Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create build directory
if (Test-Path "build_standalone") {
    Write-Host "Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build_standalone
}

New-Item -ItemType Directory -Path "build_standalone" | Out-Null
Set-Location build_standalone

Write-Host "Configuring with CMake..." -ForegroundColor Cyan

# Use the standalone CMakeLists
Copy-Item ..\CMakeLists_Standalone.txt CMakeLists.txt

cmake .. -G "Visual Studio 17 2022" -A x64

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "Building..." -ForegroundColor Cyan

cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Build Successful!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Set-Location ..

Write-Host "Running CUDA test..." -ForegroundColor Cyan
Write-Host ""

if (Test-Path "build_standalone\Release\test_cuda_simple.exe") {
    .\build_standalone\Release\test_cuda_simple.exe
} else {
    Write-Host "Test executable not found at expected location" -ForegroundColor Yellow
}

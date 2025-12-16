# Build CinematicFX Plugin with Adobe SDK
# Full production build

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CinematicFX Full Build (With Adobe SDK)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify Adobe SDK exists
$sdkPath = "C:\Users\Admin\Downloads\AfterEffectsSDK_25.6_61_win\AfterEffectsSDK_25.6_61_win\ae25.6_61.64bit.AfterEffectsSDK\Examples"

if (-not (Test-Path $sdkPath)) {
    Write-Host "ERROR: Adobe SDK not found at: $sdkPath" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Adobe SDK found: $sdkPath" -ForegroundColor Green

# Verify CUDA
$nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvccPath) {
    Write-Host "✓ CUDA Toolkit found: $($nvccPath.Source)" -ForegroundColor Green
} else {
    Write-Host "⚠ CUDA Toolkit not found - GPU acceleration will be limited" -ForegroundColor Yellow
}

Write-Host ""

# Create build directory
if (Test-Path "build") {
    Write-Host "Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
}

New-Item -ItemType Directory -Path "build" | Out-Null
Set-Location build

Write-Host "Configuring with CMake..." -ForegroundColor Cyan
Write-Host ""

cmake .. -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DADOBE_SDK_ROOT="$sdkPath"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "Building Release configuration..." -ForegroundColor Cyan
Write-Host ""

cmake --build . --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Check that Visual Studio 2022 is installed" -ForegroundColor Gray
    Write-Host "- Verify CUDA Toolkit is installed (if using GPU)" -ForegroundColor Gray
    Write-Host "- Ensure all source files are present" -ForegroundColor Gray
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Build Successful!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Find the built plugin
$pluginPath = Get-ChildItem -Path . -Filter "*.prm" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1

if ($pluginPath) {
    Write-Host "Plugin built at:" -ForegroundColor Cyan
    Write-Host "  $($pluginPath.FullName)" -ForegroundColor White
    Write-Host ""
    Write-Host "Installation:" -ForegroundColor Cyan
    Write-Host "  Copy to: C:\Program Files\Adobe\Adobe Premiere Pro\Plug-ins\Common\" -ForegroundColor Gray
} else {
    Write-Host "Plugin files:" -ForegroundColor Cyan
    Get-ChildItem -Path Release -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $($_.FullName)" -ForegroundColor White
    }
}

Write-Host ""
Set-Location ..

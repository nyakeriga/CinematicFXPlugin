# Test CUDA Installation
# Compiles and runs a simple CUDA test program

Write-Host "Testing CUDA Installation..." -ForegroundColor Cyan
Write-Host ""

# Check if nvcc is available
$nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue

if (-not $nvccPath) {
    Write-Host "ERROR: nvcc not found in PATH!" -ForegroundColor Red
    Write-Host "Please ensure CUDA Toolkit is installed and restart your terminal." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found NVCC: $($nvccPath.Source)" -ForegroundColor Green

# Get CUDA version
Write-Host ""
Write-Host "CUDA Version:" -ForegroundColor Cyan
nvcc --version

Write-Host ""
Write-Host "Compiling test program..." -ForegroundColor Cyan

# Compile the test
$compileResult = nvcc test_cuda.cpp -o test_cuda.exe 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Compilation failed!" -ForegroundColor Red
    Write-Host $compileResult
    exit 1
}

Write-Host "✓ Compilation successful!" -ForegroundColor Green

Write-Host ""
Write-Host "Running test program..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Gray

.\test_cuda.exe

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ CUDA is working correctly!" -ForegroundColor Green
} else {
    Write-Host "✗ Test failed!" -ForegroundColor Red
}

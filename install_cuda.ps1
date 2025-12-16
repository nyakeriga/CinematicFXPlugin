# CUDA Toolkit 12.0 Automatic Installation Script
# This script downloads and installs CUDA Toolkit 12.0 for Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CUDA Toolkit 12.0 Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Please right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Configuration
$cudaVersion = "12.0.0"
$cudaDownloadUrl = "https://developer.download.nvidia.com/compute/cuda/12.0.0/network_installers/cuda_12.0.0_windows_network.exe"
$installerPath = "$env:TEMP\cuda_12.0.0_installer.exe"
$installPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"

Write-Host "Checking for existing CUDA installation..." -ForegroundColor Yellow

# Check if CUDA is already installed
if (Test-Path $installPath) {
    Write-Host "CUDA 12.0 appears to be already installed at: $installPath" -ForegroundColor Green
    $response = Read-Host "Do you want to reinstall? (y/n)"
    if ($response -ne 'y') {
        Write-Host "Installation cancelled." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "Step 1: Downloading CUDA Toolkit 12.0..." -ForegroundColor Cyan
Write-Host "URL: $cudaDownloadUrl" -ForegroundColor Gray
Write-Host "This may take several minutes depending on your connection..." -ForegroundColor Gray
Write-Host ""

try {
    # Download CUDA installer
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $cudaDownloadUrl -OutFile $installerPath -UseBasicParsing
    $ProgressPreference = 'Continue'
    
    Write-Host "Download completed successfully!" -ForegroundColor Green
    Write-Host "Installer saved to: $installerPath" -ForegroundColor Gray
    Write-Host ""
    
} catch {
    Write-Host "ERROR: Failed to download CUDA Toolkit!" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Download manually from:" -ForegroundColor Yellow
    Write-Host "https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
    exit 1
}

Write-Host "Step 2: Installing CUDA Toolkit 12.0..." -ForegroundColor Cyan
Write-Host "This will take 5-10 minutes. Please wait..." -ForegroundColor Gray
Write-Host ""

try {
    # Run installer silently
    $installArgs = @(
        "-s",  # Silent installation
        "nvcc_12.0",  # CUDA Compiler
        "cudart_12.0",  # CUDA Runtime
        "cublas_12.0",  # CUDA BLAS Library
        "cublas_dev_12.0",  # CUDA BLAS Development
        "curand_12.0",  # CUDA Random Number Generation
        "curand_dev_12.0",
        "cusparse_12.0",  # CUDA Sparse Matrix
        "cusparse_dev_12.0",
        "npp_12.0",  # CUDA NPP Library
        "npp_dev_12.0",
        "nvrtc_12.0",  # CUDA Runtime Compilation
        "nvrtc_dev_12.0"
    )
    
    $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        Write-Host "CUDA Toolkit installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Installer returned exit code: $($process.ExitCode)" -ForegroundColor Yellow
        Write-Host "Installation may have completed with warnings." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "ERROR: Installation failed!" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3: Cleaning up..." -ForegroundColor Cyan

# Clean up installer
if (Test-Path $installerPath) {
    Remove-Item $installerPath -Force
    Write-Host "Installer removed." -ForegroundColor Gray
}

Write-Host ""
Write-Host "Step 4: Verifying installation..." -ForegroundColor Cyan

# Set environment variables
$cudaBinPath = "$installPath\bin"
$cudaLibPath = "$installPath\lib\x64"

if (Test-Path $cudaBinPath) {
    Write-Host "✓ CUDA binaries found at: $cudaBinPath" -ForegroundColor Green
} else {
    Write-Host "✗ CUDA binaries not found!" -ForegroundColor Red
}

# Check for nvcc (CUDA compiler)
$nvccPath = "$cudaBinPath\nvcc.exe"
if (Test-Path $nvccPath) {
    Write-Host "✓ NVCC compiler found" -ForegroundColor Green
    
    # Try to get version
    try {
        $version = & $nvccPath --version 2>&1 | Select-String -Pattern "release"
        Write-Host "  Version: $version" -ForegroundColor Gray
    } catch {
        Write-Host "  (Unable to query version)" -ForegroundColor Gray
    }
} else {
    Write-Host "✗ NVCC compiler not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "Step 5: Configuring environment variables..." -ForegroundColor Cyan

# Add to PATH if not already present
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$pathsToAdd = @($cudaBinPath, $cudaLibPath)

foreach ($pathToAdd in $pathsToAdd) {
    if ($currentPath -notlike "*$pathToAdd*") {
        Write-Host "Adding to PATH: $pathToAdd" -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$currentPath;$pathToAdd",
            "Machine"
        )
    } else {
        Write-Host "Already in PATH: $pathToAdd" -ForegroundColor Gray
    }
}

# Set CUDA_PATH
[Environment]::SetEnvironmentVariable("CUDA_PATH", $installPath, "Machine")
[Environment]::SetEnvironmentVariable("CUDA_PATH_V12_0", $installPath, "Machine")

Write-Host "Environment variables configured." -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: You must restart your terminal/IDE for environment variables to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "CUDA Toolkit 12.0 installed at: $installPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Close and restart Visual Studio Code" -ForegroundColor White
Write-Host "2. Open a new PowerShell terminal" -ForegroundColor White
Write-Host "3. Verify installation with: nvcc --version" -ForegroundColor White
Write-Host ""

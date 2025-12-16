/*******************************************************************************
 * CinematicFX - CUDA Test Program
 * 
 * Tests CUDA installation and basic functionality
 ******************************************************************************/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "========================================\n";
    std::cout << "CinematicFX CUDA Test\n";
    std::cout << "========================================\n\n";
    
    // Check CUDA device count
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "ERROR: Failed to query CUDA devices!\n";
        std::cerr << "Error: " << cudaGetErrorString(error) << "\n";
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "ERROR: No CUDA-capable devices found!\n";
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)\n\n";
    
    // Display device information
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Grid Size: " 
                  << prop.maxGridSize[0] << " x " 
                  << prop.maxGridSize[1] << " x " 
                  << prop.maxGridSize[2] << "\n";
        std::cout << "\n";
    }
    
    // Simple memory test
    std::cout << "Testing CUDA memory operations...\n";
    
    const int N = 1024;
    float* d_data = nullptr;
    
    error = cudaMalloc(&d_data, N * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed!\n";
        std::cerr << "Error: " << cudaGetErrorString(error) << "\n";
        return 1;
    }
    
    std::cout << "✓ Successfully allocated " << N * sizeof(float) << " bytes on GPU\n";
    
    cudaFree(d_data);
    std::cout << "✓ Successfully freed GPU memory\n";
    
    std::cout << "\n========================================\n";
    std::cout << "CUDA Test: PASSED\n";
    std::cout << "========================================\n";
    
    return 0;
}

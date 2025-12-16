/*******************************************************************************
 * CinematicFX - CUDA Halation Kernels
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Extract extreme highlights (red channel for film halation)
__global__ void halation_extract_kernel(const float* input, float* output,
                                        int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    
    float r = input[idx + 0];
    float g = input[idx + 1];
    float b = input[idx + 2];
    float a = input[idx + 3];
    
    // Luminance
    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    
    // Only extract extreme highlights
    if (luma > threshold) {
        // Create red fringe (halation is primarily red on film)
        output[idx + 0] = r;  // Full red channel
        output[idx + 1] = 0.0f;
        output[idx + 2] = 0.0f;
        output[idx + 3] = a;
    } else {
        output[idx + 0] = 0.0f;
        output[idx + 1] = 0.0f;
        output[idx + 2] = 0.0f;
        output[idx + 3] = a;
    }
}

// Blur halation fringe
__global__ void halation_blur_kernel(const float* input, float* output,
                                     int width, int height, float radius,
                                     bool horizontal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernel_size = int(radius * 2.0f) + 1;
    float sigma = radius / 3.0f;
    
    float4 sum = make_float4(0, 0, 0, 0);
    float weight_sum = 0.0f;
    
    for (int i = -kernel_size; i <= kernel_size; ++i) {
        int sample_x = horizontal ? (x + i) : x;
        int sample_y = horizontal ? y : (y + i);
        
        if (sample_x >= 0 && sample_x < width && sample_y >= 0 && sample_y < height) {
            int sample_idx = (sample_y * width + sample_x) * 4;
            
            float distance = abs(float(i));
            float weight = expf(-(distance * distance) / (2.0f * sigma * sigma));
            
            sum.x += input[sample_idx + 0] * weight;
            sum.y += input[sample_idx + 1] * weight;
            sum.z += input[sample_idx + 2] * weight;
            sum.w += input[sample_idx + 3] * weight;
            weight_sum += weight;
        }
    }
    
    int idx = (y * width + x) * 4;
    output[idx + 0] = sum.x / weight_sum;
    output[idx + 1] = sum.y / weight_sum;
    output[idx + 2] = sum.z / weight_sum;
    output[idx + 3] = sum.w / weight_sum;
}

// Blend halation fringe with original
__global__ void halation_blend_kernel(const float* original, const float* halation,
                                      float* output, int width, int height,
                                      float intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    
    // Additive blend with red fringe
    output[idx + 0] = original[idx + 0] + halation[idx + 0] * intensity;
    output[idx + 1] = original[idx + 1] + halation[idx + 1] * intensity;
    output[idx + 2] = original[idx + 2] + halation[idx + 2] * intensity;
    output[idx + 3] = original[idx + 3];
}

// Host launchers
extern "C" {

void halation_extract_cuda(const float* input, float* output, int width, int height,
                           float threshold, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    halation_extract_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold);
}

void halation_blur_cuda(const float* input, float* output, int width, int height,
                        float radius, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    // Two-pass separable blur
    float* temp;
    cudaMalloc(&temp, width * height * 4 * sizeof(float));
    
    halation_blur_kernel<<<grid, block, 0, stream>>>(input, temp, width, height, radius, true);
    halation_blur_kernel<<<grid, block, 0, stream>>>(temp, output, width, height, radius, false);
    
    cudaFree(temp);
}

void halation_blend_cuda(const float* original, const float* halation, float* output,
                         int width, int height, float intensity, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    halation_blend_kernel<<<grid, block, 0, stream>>>(original, halation, output, width, height, intensity);
}

} // extern "C"

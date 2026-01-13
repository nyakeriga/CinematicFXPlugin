/*******************************************************************************
 * CinematicFX - CUDA Glow Kernels
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Extract highlights above threshold
__global__ void glow_extract_kernel(const float* input, float* output,
                                    int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    
    float r = input[idx + 0];
    float g = input[idx + 1];
    float b = input[idx + 2];
    float a = input[idx + 3];
    
    // Luminance (Rec.709)
    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    
    // Extract if above threshold, with smooth falloff
    if (luma > threshold) {
        float scale = (luma - threshold) / (1.0f - threshold + 0.001f);
        scale = powf(scale, 0.8f); // Softer falloff for more natural glow
        output[idx + 0] = r * scale;
        output[idx + 1] = g * scale;
        output[idx + 2] = b * scale;
        output[idx + 3] = a;
    } else {
        output[idx + 0] = 0.0f;
        output[idx + 1] = 0.0f;
        output[idx + 2] = 0.0f;
        output[idx + 3] = a;
    }
}

// Large Gaussian blur for diffusion effect
__global__ void glow_blur_kernel(const float* input, float* output,
                                 int width, int height, float radius,
                                 bool horizontal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernel_size = min(int(radius * 2.0f) + 1, 16); // Limit for performance
    float sigma = radius / 2.0f;
    
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

// Blend glow with original (with tint, desaturation, blend modes)
__global__ void glow_blend_kernel(const float* original, const float* glow,
                                  float* output, int width, int height,
                                  float intensity, float desaturation,
                                  int blend_mode, float tint_r, float tint_g, float tint_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    // Get original and glow colors
    float orig_r = original[idx + 0];
    float orig_g = original[idx + 1];
    float orig_b = original[idx + 2];
    float orig_a = original[idx + 3];

    float glow_r = glow[idx + 0];
    float glow_g = glow[idx + 1];
    float glow_b = glow[idx + 2];

    // Apply desaturation to glow
    if (desaturation > 0.0f) {
        float glow_luma = 0.2126f * glow_r + 0.7152f * glow_g + 0.0722f * glow_b;
        glow_r = glow_r * (1.0f - desaturation) + glow_luma * desaturation;
        glow_g = glow_g * (1.0f - desaturation) + glow_luma * desaturation;
        glow_b = glow_b * (1.0f - desaturation) + glow_luma * desaturation;
    }

    // Apply tint to glow
    glow_r *= tint_r;
    glow_g *= tint_g;
    glow_b *= tint_b;

    // Apply blend mode
    float final_r, final_g, final_b;
    if (blend_mode == 0) { // Screen
        final_r = 1.0f - (1.0f - orig_r) * (1.0f - glow_r * intensity);
        final_g = 1.0f - (1.0f - orig_g) * (1.0f - glow_g * intensity);
        final_b = 1.0f - (1.0f - orig_b) * (1.0f - glow_b * intensity);
    } else if (blend_mode == 1) { // Add
        final_r = orig_r + glow_r * intensity;
        final_g = orig_g + glow_g * intensity;
        final_b = orig_b + glow_b * intensity;
    } else { // Normal (overlay-like)
        final_r = orig_r + glow_r * intensity;
        final_g = orig_g + glow_g * intensity;
        final_b = orig_b + glow_b * intensity;
    }

    output[idx + 0] = final_r;
    output[idx + 1] = final_g;
    output[idx + 2] = final_b;
    output[idx + 3] = orig_a;
}

// Host launchers
extern "C" {

void glow_extract_cuda(const float* input, float* output, int width, int height,
                       float threshold, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    glow_extract_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold);
}

void glow_blur_cuda(const float* input, float* output, int width, int height,
                    float radius_x, float radius_y, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // Two-pass separable blur
    float* temp;
    cudaMalloc(&temp, width * height * 4 * sizeof(float));

    glow_blur_kernel<<<grid, block, 0, stream>>>(input, temp, width, height, radius_x, true);
    glow_blur_kernel<<<grid, block, 0, stream>>>(temp, output, width, height, radius_y, false);

    cudaFree(temp);
}

void glow_blend_cuda(const float* original, const float* glow, float* output,
                     int width, int height, float intensity, float desaturation,
                     int blend_mode, float tint_r, float tint_g, float tint_b, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    glow_blend_kernel<<<grid, block, 0, stream>>>(original, glow, output, width, height,
                                                   intensity, desaturation, blend_mode,
                                                   tint_r, tint_g, tint_b);
}

} // extern "C"

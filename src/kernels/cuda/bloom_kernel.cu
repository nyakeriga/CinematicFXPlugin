/*******************************************************************************
 * CinematicFX - CUDA Bloom Kernel
 * 
 * GPU-accelerated Bloom effect (atmospheric diffusion)
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Bloom parameters structure (must match C++ header)
struct BloomParams {
    float amount;
    float radius;
    float tint_r;
    float tint_g;
    float tint_b;
    float shadow_boost;
};

// Texture references
texture<float4, cudaTextureType2D, cudaReadModeElementType> input_tex;

/**
 * @brief Calculate luminance (Rec. 709)
 */
__device__ inline float luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

/**
 * @brief Shadow/midtone lift curve
 */
__device__ inline float lift_curve(float luma, float boost) {
    return powf(1.0f - luma, 2.2f) * boost;
}

/**
 * @brief Kernel: Extract luminance and apply shadow lift
 */
__global__ void bloom_extract_kernel(
    float4* output,
    int width,
    int height,
    BloomParams params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sample input texture
    float4 color = tex2D(input_tex, x + 0.5f, y + 0.5f);
    
    // Calculate luminance
    float luma = luminance(color.x, color.y, color.z);
    
    // Apply shadow/midtone lift
    float boost = lift_curve(luma, params.shadow_boost);
    
    color.x += boost;
    color.y += boost;
    color.z += boost;
    
    // Write to output
    output[y * width + x] = color;
}

/**
 * @brief Kernel: Horizontal Gaussian blur
 */
__global__ void bloom_blur_horizontal_kernel(
    const float4* input,
    float4* output,
    const float* kernel,
    int kernel_size,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float4 sum = make_float4(0, 0, 0, 0);
    int half_kernel = kernel_size / 2;
    
    // Convolve horizontally
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_x = min(max(x + i, 0), width - 1);
        float4 pixel = input[y * width + sample_x];
        float weight = kernel[i + half_kernel];
        
        sum.x += pixel.x * weight;
        sum.y += pixel.y * weight;
        sum.z += pixel.z * weight;
        sum.w += pixel.w * weight;
    }
    
    output[y * width + x] = sum;
}

/**
 * @brief Kernel: Vertical Gaussian blur
 */
__global__ void bloom_blur_vertical_kernel(
    const float4* input,
    float4* output,
    const float* kernel,
    int kernel_size,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float4 sum = make_float4(0, 0, 0, 0);
    int half_kernel = kernel_size / 2;
    
    // Convolve vertically
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_y = min(max(y + i, 0), height - 1);
        float4 pixel = input[sample_y * width + x];
        float weight = kernel[i + half_kernel];
        
        sum.x += pixel.x * weight;
        sum.y += pixel.y * weight;
        sum.z += pixel.z * weight;
        sum.w += pixel.w * weight;
    }
    
    output[y * width + x] = sum;
}

/**
 * @brief Kernel: Additive blend with tint
 */
__global__ void bloom_blend_kernel(
    const float4* base,
    const float4* bloom,
    float4* output,
    int width,
    int height,
    BloomParams params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float4 base_color = base[idx];
    float4 bloom_color = bloom[idx];
    
    // Apply tint to bloom
    bloom_color.x *= params.tint_r;
    bloom_color.y *= params.tint_g;
    bloom_color.z *= params.tint_b;
    
    // Additive blend
    float4 result;
    result.x = base_color.x + bloom_color.x * params.amount;
    result.y = base_color.y + bloom_color.y * params.amount;
    result.z = base_color.z + bloom_color.z * params.amount;
    result.w = base_color.w;
    
    output[idx] = result;
}

// Host functions (called from C++)
extern "C" {

void launch_bloom_extract(
    float4* output,
    cudaArray* input_array,
    int width,
    int height,
    BloomParams params
) {
    // Bind texture
    cudaBindTextureToArray(input_tex, input_array);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bloom_extract_kernel<<<grid, block>>>(output, width, height, params);
    
    cudaUnbindTexture(input_tex);
}

void launch_bloom_blur_horizontal(
    const float4* input,
    float4* output,
    const float* kernel,
    int kernel_size,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bloom_blur_horizontal_kernel<<<grid, block>>>(
        input, output, kernel, kernel_size, width, height
    );
}

void launch_bloom_blur_vertical(
    const float4* input,
    float4* output,
    const float* kernel,
    int kernel_size,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bloom_blur_vertical_kernel<<<grid, block>>>(
        input, output, kernel, kernel_size, width, height
    );
}

void launch_bloom_blend(
    const float4* base,
    const float4* bloom,
    float4* output,
    int width,
    int height,
    BloomParams params
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bloom_blend_kernel<<<grid, block>>>(
        base, bloom, output, width, height, params
    );
}

} // extern "C"

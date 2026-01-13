/*******************************************************************************
 * CinematicFX - CUDA Halation Kernels
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// HSL to RGB conversion
__device__ void hsl_to_rgb(float h, float s, float l, float* r, float* g, float* b) {
    h = fmodf(h, 360.0f);
    if (h < 0.0f) h += 360.0f;
    h /= 360.0f;

    float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = l - c / 2.0f;

    if (h < 1.0f/6.0f) {
        *r = c; *g = x; *b = 0.0f;
    } else if (h < 2.0f/6.0f) {
        *r = x; *g = c; *b = 0.0f;
    } else if (h < 3.0f/6.0f) {
        *r = 0.0f; *g = c; *b = x;
    } else if (h < 4.0f/6.0f) {
        *r = 0.0f; *g = x; *b = c;
    } else if (h < 5.0f/6.0f) {
        *r = x; *g = 0.0f; *b = c;
    } else {
        *r = c; *g = 0.0f; *b = x;
    }

    *r += m;
    *g += m;
    *b += m;
}

// Extract extreme highlights with color control
__global__ void halation_extract_kernel(const float* input, float* output,
                                        int width, int height, float threshold,
                                        float hue, float saturation) {
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
        // Use HSL to RGB for precise color control
        float fringe_r, fringe_g, fringe_b;
        hsl_to_rgb(hue, saturation, 0.5f, &fringe_r, &fringe_g, &fringe_b);

        // Scale by extracted brightness with smooth falloff
        float scale = (luma - threshold) / (1.0f - threshold + 0.001f);
        scale = powf(scale, 0.7f); // Softer falloff for film-like halation
        output[idx + 0] = fringe_r * scale;
        output[idx + 1] = fringe_g * scale;
        output[idx + 2] = fringe_b * scale;
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
                           float threshold, float hue, float saturation, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    halation_extract_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold, hue, saturation);
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

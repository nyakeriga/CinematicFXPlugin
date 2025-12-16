/*******************************************************************************
 * CinematicFX - CUDA Chromatic Aberration Kernel
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Bilinear texture sampling
__device__ float4 sample_bilinear(const float* texture, int width, int height,
                                   float u, float v) {
    // Clamp coordinates
    u = fminf(fmaxf(u, 0.0f), float(width - 1));
    v = fminf(fmaxf(v, 0.0f), float(height - 1));
    
    int x0 = int(floorf(u));
    int y0 = int(floorf(v));
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float fx = u - float(x0);
    float fy = v - float(y0);
    
    // Sample four pixels
    int idx00 = (y0 * width + x0) * 4;
    int idx10 = (y0 * width + x1) * 4;
    int idx01 = (y1 * width + x0) * 4;
    int idx11 = (y1 * width + x1) * 4;
    
    float4 p00 = make_float4(texture[idx00], texture[idx00 + 1], texture[idx00 + 2], texture[idx00 + 3]);
    float4 p10 = make_float4(texture[idx10], texture[idx10 + 1], texture[idx10 + 2], texture[idx10 + 3]);
    float4 p01 = make_float4(texture[idx01], texture[idx01 + 1], texture[idx01 + 2], texture[idx01 + 3]);
    float4 p11 = make_float4(texture[idx11], texture[idx11 + 1], texture[idx11 + 2], texture[idx11 + 3]);
    
    // Bilinear interpolation
    float4 top = make_float4(
        p00.x * (1.0f - fx) + p10.x * fx,
        p00.y * (1.0f - fx) + p10.y * fx,
        p00.z * (1.0f - fx) + p10.z * fx,
        p00.w * (1.0f - fx) + p10.w * fx
    );
    
    float4 bottom = make_float4(
        p01.x * (1.0f - fx) + p11.x * fx,
        p01.y * (1.0f - fx) + p11.y * fx,
        p01.z * (1.0f - fx) + p11.z * fx,
        p01.w * (1.0f - fx) + p11.w * fx
    );
    
    return make_float4(
        top.x * (1.0f - fy) + bottom.x * fy,
        top.y * (1.0f - fy) + bottom.y * fy,
        top.z * (1.0f - fy) + bottom.z * fy,
        top.w * (1.0f - fy) + bottom.w * fy
    );
}

// Apply chromatic aberration
__global__ void chromatic_aberration_kernel(const float* input, float* output,
                                            int width, int height,
                                            float amount, float angle_deg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Center coordinates
    float center_x = float(width) * 0.5f;
    float center_y = float(height) * 0.5f;
    
    // Vector from center
    float dx = float(x) - center_x;
    float dy = float(y) - center_y;
    
    // Distance from center (normalized)
    float distance = sqrtf(dx * dx + dy * dy) / sqrtf(center_x * center_x + center_y * center_y);
    
    // Convert angle to radians
    float angle_rad = angle_deg * 3.14159265f / 180.0f;
    
    // Calculate directional offset
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);
    
    // Radial offset scaled by distance from center
    float offset = amount * distance * 2.0f;
    
    // Sample each channel with offset
    float offset_x = cos_a * offset;
    float offset_y = sin_a * offset;
    
    // Red channel: offset in positive direction
    float4 red_sample = sample_bilinear(input, width, height,
                                        float(x) + offset_x,
                                        float(y) + offset_y);
    
    // Green channel: no offset
    int idx_green = (y * width + x) * 4;
    float g = input[idx_green + 1];
    
    // Blue channel: offset in negative direction
    float4 blue_sample = sample_bilinear(input, width, height,
                                         float(x) - offset_x,
                                         float(y) - offset_y);
    
    // Combine channels
    int idx = (y * width + x) * 4;
    output[idx + 0] = red_sample.x;   // Red from offset sample
    output[idx + 1] = g;              // Green from center
    output[idx + 2] = blue_sample.z;  // Blue from offset sample
    output[idx + 3] = input[idx + 3]; // Alpha unchanged
}

// Host launcher
extern "C" {

void chromatic_aberration_apply_cuda(const float* input, float* output,
                                      int width, int height,
                                      float amount, float angle,
                                      cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    chromatic_aberration_kernel<<<grid, block, 0, stream>>>(input, output, width, height,
                                                              amount, angle);
}

} // extern "C"

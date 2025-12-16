/*******************************************************************************
 * CinematicFX - Metal Bloom Shader
 * 
 * GPU-accelerated Bloom effect (atmospheric diffusion) for Apple Metal
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// Bloom parameters structure (must match C++ header)
struct BloomParams {
    float amount;
    float radius;
    float tint_r;
    float tint_g;
    float tint_b;
    float shadow_boost;
};

/**
 * @brief Calculate luminance (Rec. 709)
 */
inline float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

/**
 * @brief Shadow/midtone lift curve
 */
inline float lift_curve(float luma, float boost) {
    return pow(1.0 - luma, 2.2) * boost;
}

/**
 * @brief Shader: Extract luminance and apply shadow lift
 */
kernel void bloom_extract(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BloomParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    // Sample input
    float4 color = input.read(gid);
    
    // Calculate luminance
    float luma = luminance(color.rgb);
    
    // Apply shadow/midtone lift
    float boost = lift_curve(luma, params.shadow_boost);
    
    color.rgb += boost;
    
    // Write output
    output.write(color, gid);
}

/**
 * @brief Shader: Horizontal Gaussian blur
 */
kernel void bloom_blur_horizontal(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* kernel [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    int width = input.get_width();
    
    // Convolve horizontally
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_x = clamp(int(gid.x) + i, 0, width - 1);
        float4 pixel = input.read(uint2(sample_x, gid.y));
        float weight = kernel[i + half_kernel];
        
        sum += pixel * weight;
    }
    
    output.write(sum, gid);
}

/**
 * @brief Shader: Vertical Gaussian blur
 */
kernel void bloom_blur_vertical(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* kernel [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    int height = input.get_height();
    
    // Convolve vertically
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_y = clamp(int(gid.y) + i, 0, height - 1);
        float4 pixel = input.read(uint2(gid.x, sample_y));
        float weight = kernel[i + half_kernel];
        
        sum += pixel * weight;
    }
    
    output.write(sum, gid);
}

/**
 * @brief Shader: Additive blend with tint
 */
kernel void bloom_blend(
    texture2d<float, access::read> base [[texture(0)]],
    texture2d<float, access::read> bloom [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant BloomParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 base_color = base.read(gid);
    float4 bloom_color = bloom.read(gid);
    
    // Apply tint to bloom
    bloom_color.rgb *= float3(params.tint_r, params.tint_g, params.tint_b);
    
    // Additive blend
    float4 result = base_color;
    result.rgb += bloom_color.rgb * params.amount;
    
    output.write(result, gid);
}

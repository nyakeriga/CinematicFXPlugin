/*******************************************************************************
 * CinematicFX - Comprehensive Metal Shader Pipeline
 * 
 * All cinematic effects in one Metal shader library for Apple Metal
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// Parameter structures for all effects
struct BloomParams {
    float amount;
    float radius;
    float tint_r;
    float tint_g;
    float tint_b;
    float shadow_boost;
};

struct GlowParams {
    float threshold;           // Luminance threshold: 0.0 - 1.0
    float radius_x;            // Horizontal radius: 1.0 - 100.0
    float radius_y;            // Vertical radius: 1.0 - 100.0
    float intensity;           // Glow strength: 0.0 - 2.0
    float desaturation;        // Color desaturation: 0.0 - 1.0
    int blend_mode;            // 0=Screen, 1=Add, 2=Normal
    float tint_r;              // Tint red component: 0.0 - 1.0
    float tint_g;              // Tint green component: 0.0 - 1.0
    float tint_b;              // Tint blue component: 0.0 - 1.0
};

struct HalationParams {
    float intensity;           // Fringe strength: 0.0 - 1.0
    float spread;              // Fringe spread: 1.0 - 50.0
};

struct GrainParams {
    float shadows_amount;      // Shadows grain intensity: 0.0 - 1.0
    float mids_amount;         // Midtones grain intensity: 0.0 - 1.0
    float highlights_amount;   // Highlights grain intensity: 0.0 - 1.0
    float size;                // Grain texture scale: 0.5 - 5.0
    float roughness;           // Grain distribution: 0.0 - 1.0
    float saturation;          // Grain color saturation: 0.0 - 2.0
};

struct ChromaticAberrationParams {
    float amount;              // Overall aberration intensity: 0.0 - 1.0
    float red_scale;           // Red channel scale: 0.5 - 2.0
    float green_scale;         // Green channel scale: 0.5 - 2.0
    float blue_scale;          // Blue channel scale: 0.5 - 2.0
    float blurriness;          // Edge softness: 0.0 - 10.0
    float angle;               // Offset direction in degrees: 0.0 - 360.0
};

/**
 * @brief Utility functions
 */
inline float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

inline float shadow_lift_curve(float luma, float boost) {
    return pow(1.0 - luma, 2.2) * boost;
}

inline float3 desaturate(float3 color, float amount) {
    float gray = luminance(color);
    return mix(color, float3(gray), amount);
}

inline float3 blend_screen(float3 base, float3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

inline float3 blend_add(float3 base, float3 blend, float intensity) {
    return min(base + blend * intensity, 1.0);
}

inline float3 blend_normal(float3 base, float3 blend, float intensity) {
    return mix(base, blend, intensity);
}

inline float hash(float2 p, float seed) {
    return fract(sin(dot(p, float2(12.9898, 78.233)) + seed) * 43758.5453);
}

inline float noise(float2 p, float seed) {
    float2 i = floor(p);
    float2 f = fract(p);
    
    float a = hash(i, seed);
    float b = hash(i + float2(1.0, 0.0), seed);
    float c = hash(i + float2(0.0, 1.0), seed);
    float d = hash(i + float2(1.0, 1.0), seed);
    
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// =============================================================================
// BLOOM SHADERS
// =============================================================================

kernel void bloom_extract(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BloomParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 color = input.read(gid);
    float luma = luminance(color.rgb);
    float boost = shadow_lift_curve(luma, params.shadow_boost);
    color.rgb += boost;
    output.write(color, gid);
}

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
    
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_x = clamp(int(gid.x) + i, 0, width - 1);
        float4 pixel = input.read(uint2(sample_x, gid.y));
        float weight = kernel[i + half_kernel];
        sum += pixel * weight;
    }
    
    output.write(sum, gid);
}

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
    
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_y = clamp(int(gid.y) + i, 0, height - 1);
        float4 pixel = input.read(uint2(gid.x, sample_y));
        float weight = kernel[i + half_kernel];
        sum += pixel * weight;
    }
    
    output.write(sum, gid);
}

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
    bloom_color.rgb *= float3(params.tint_r, params.tint_g, params.tint_b);
    
    float4 result = base_color;
    result.rgb += bloom_color.rgb * params.amount;
    output.write(result, gid);
}

// =============================================================================
// GLOW SHADERS (Enhanced with separate X/Y radius and blend modes)
// =============================================================================

kernel void glow_extract(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant GlowParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 color = input.read(gid);
    float luma = luminance(color.rgb);
    float highlight = smoothstep(params.threshold, 1.0, luma);
    float3 processed_color = desaturate(color.rgb, params.desaturation);
    
    float4 result = float4(processed_color * highlight, color.a);
    output.write(result, gid);
}

kernel void glow_blur_horizontal(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* kernel [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    constant GlowParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    int width = input.get_width();
    float scale = params.radius_x / 10.0;
    
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_x = clamp(int(gid.x) + int(i * scale), 0, width - 1);
        float4 pixel = input.read(uint2(sample_x, gid.y));
        float weight = kernel[i + half_kernel];
        sum += pixel * weight;
    }
    
    output.write(sum, gid);
}

kernel void glow_blur_vertical(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* kernel [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    constant GlowParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    int height = input.get_height();
    float scale = params.radius_y / 10.0;
    
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int sample_y = clamp(int(gid.y) + int(i * scale), 0, height - 1);
        float4 pixel = input.read(uint2(gid.x, sample_y));
        float weight = kernel[i + half_kernel];
        sum += pixel * weight;
    }
    
    output.write(sum, gid);
}

kernel void glow_blend(
    texture2d<float, access::read> base [[texture(0)]],
    texture2d<float, access::read> glow [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant GlowParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 base_color = base.read(gid);
    float4 glow_color = glow.read(gid);
    glow_color.rgb *= float3(params.tint_r, params.tint_g, params.tint_b);
    
    float3 blended;
    switch (params.blend_mode) {
        case 0: // Screen
            blended = blend_screen(base_color.rgb, glow_color.rgb);
            break;
        case 1: // Add
            blended = blend_add(base_color.rgb, glow_color.rgb, params.intensity);
            break;
        case 2: // Normal
        default:
            blended = blend_normal(base_color.rgb, glow_color.rgb, params.intensity);
            break;
    }
    
    float4 result = base_color;
    result.rgb = blended;
    output.write(result, gid);
}

// =============================================================================
// HALATION SHADERS (Film fringe effect)
// =============================================================================

kernel void halation_blur(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* kernel [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    constant HalationParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    int width = input.get_width();
    int height = input.get_height();
    float scale = params.spread / 5.0;
    
    // Large radius blur for halation effect
    for (int y = -half_kernel; y <= half_kernel; y++) {
        for (int x = -half_kernel; x <= half_kernel; x++) {
            int sample_x = clamp(int(gid.x) + int(x * scale), 0, width - 1);
            int sample_y = clamp(int(gid.y) + int(y * scale), 0, height - 1);
            float4 pixel = input.read(uint2(sample_x, sample_y));
            float weight = kernel[x + half_kernel] * kernel[y + half_kernel];
            sum += pixel * weight;
        }
    }
    
    output.write(sum, gid);
}

kernel void halation_apply(
    texture2d<float, access::read> base [[texture(0)]],
    texture2d<float, access::read> halation [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant HalationParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 base_color = base.read(gid);
    float4 halation_color = halation.read(gid);
    
    // Apply red fringe to highlights
    float luma = luminance(base_color.rgb);
    float highlight = smoothstep(0.7, 1.0, luma);
    
    float3 fringe = float3(1.0, 0.3, 0.1) * params.intensity * highlight;
    float3 result = base_color.rgb + halation_color.rgb * fringe;
    
    output.write(float4(result, base_color.a), gid);
}

// =============================================================================
// GRAIN SHADERS (Luminosity-based grain mapping)
// =============================================================================

kernel void grain_apply(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant GrainParams& params [[buffer(0)]],
    constant uint& frame_number [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float4 color = input.read(gid);
    float luma = luminance(color.rgb);
    
    // Generate stable grain based on position and frame
    float2 pos = float2(gid) / params.size;
    float grain_value = noise(pos, frame_number * 0.01 + luma * 10.0);
    
    // Apply luminosity-based grain amounts
    float grain_amount;
    if (luma < 0.3) {
        grain_amount = params.shadows_amount;
    } else if (luma < 0.7) {
        grain_amount = params.mids_amount;
    } else {
        grain_amount = params.highlights_amount;
    }
    
    // Apply grain with roughness variation
    float rough_grain = mix(grain_value, noise(pos * 2.0, frame_number * 0.01), params.roughness);
    float grain_noise = (rough_grain - 0.5) * 2.0 * grain_amount;
    
    // Apply grain and saturation
    float3 grained = color.rgb + grain_noise;
    grained = mix(grained, float3(luminance(grained)), 1.0 - params.saturation);
    
    output.write(float4(grained, color.a), gid);
}

// =============================================================================
// CHROMATIC ABERRATION SHADERS (RGB channel scaling)
// =============================================================================

kernel void chromatic_aberration(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant ChromaticAberrationParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    float width = float(output.get_width());
    float height = float(output.get_height());
    float2 center = float2(width * 0.5, height * 0.5);
    float2 pos = float2(gid);
    
    // Calculate offset direction
    float angle_rad = radians(params.angle);
    float2 direction = float2(cos(angle_rad), sin(angle_rad));
    
    // Calculate distance from center for scaling
    float2 to_center = pos - center;
    float distance = length(to_center);
    float normalized_distance = distance / max(width, height);
    
    // Apply RGB channel offsets with amount scaling
    float2 offset = direction * normalized_distance * params.blurriness * params.amount;
    
    // Sample each channel with different offsets
    float2 red_pos = pos + offset * params.red_scale;
    float2 green_pos = pos + offset * params.green_scale;
    float2 blue_pos = pos + offset * params.blue_scale;
    
    // Clamp positions
    red_pos = clamp(red_pos, float2(0.0), float2(width - 1, height - 1));
    green_pos = clamp(green_pos, float2(0.0), float2(width - 1, height - 1));
    blue_pos = clamp(blue_pos, float2(0.0), float2(width - 1, height - 1));
    
    // Sample original color
    float4 original = input.read(uint2(pos));
    
    // Sample individual channels
    float4 red_sample = input.read(uint2(red_pos));
    float4 green_sample = input.read(uint2(green_pos));
    float4 blue_sample = input.read(uint2(blue_pos));
    
    // Reconstruct with chromatic aberration
    float4 result = original;
    
    // Apply chromatic aberration with amount scaling
    if (params.amount > 0.0) {
        result.r = mix(original.r, red_sample.r, params.amount);
        result.g = mix(original.g, green_sample.g, params.amount);
        result.b = mix(original.b, blue_sample.b, params.amount);
    }
    
    output.write(result, gid);
}

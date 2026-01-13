/*******************************************************************************
 * CinematicFX - CPU Fallback Implementation
 * 
 * Software rendering fallback with SIMD optimization
 ******************************************************************************/

#include "CPUFallback.h"
#include "../utils/Logger.h"
#include "../utils/MathUtils.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace CinematicFX {

CPUFallback::CPUFallback()
    : initialized_(false)
{
}

CPUFallback::~CPUFallback() {
    Shutdown();
}

bool CPUFallback::Initialize() {
    if (initialized_) {
        return true;
    }
    
    Logger::Info("CPUFallback: Initializing software rendering backend");
    initialized_ = true;
    return true;
}

void CPUFallback::Shutdown() {
    if (!initialized_) {
        return;
    }
    
    // Release all allocated textures
    for (auto* texture : allocated_textures_) {
        if (texture && texture->data) {
            delete[] texture->data;
        }
        delete texture;
    }
    allocated_textures_.clear();
    
    initialized_ = false;
    Logger::Info("CPUFallback: Shutdown complete");
}

const char* CPUFallback::GetDeviceName() const {
    return "CPU (Software Fallback)";
}

GPUTexture CPUFallback::UploadTexture(const FrameBuffer& buffer) {
    auto* texture = new CPUTexture();
    texture->width = buffer.width;
    texture->height = buffer.height;
    texture->stride = buffer.width * 4; // RGBA
    
    size_t data_size = texture->stride * texture->height * sizeof(float);
    texture->data = new float[texture->stride * texture->height];
    
    // Copy data
    if (buffer.stride == texture->stride) {
        memcpy(texture->data, buffer.data, data_size);
    } else {
        // Copy row by row
        for (uint32_t y = 0; y < buffer.height; y++) {
            memcpy(texture->data + y * texture->stride,
                   buffer.data + y * buffer.stride,
                   buffer.width * 4 * sizeof(float));
        }
    }
    
    allocated_textures_.push_back(texture);
    return reinterpret_cast<GPUTexture>(texture);
}

bool CPUFallback::DownloadTexture(GPUTexture texture, FrameBuffer& buffer) {
    auto* cpu_texture = GetCPUTexture(texture);
    if (!cpu_texture) {
        Logger::Error("CPUFallback: Invalid texture handle in DownloadTexture");
        return false;
    }
    
    // Copy data back
    if (buffer.stride == cpu_texture->stride) {
        size_t data_size = cpu_texture->stride * cpu_texture->height * sizeof(float);
        memcpy(buffer.data, cpu_texture->data, data_size);
    } else {
        // Copy row by row
        for (uint32_t y = 0; y < buffer.height; y++) {
            memcpy(buffer.data + y * buffer.stride,
                   cpu_texture->data + y * cpu_texture->stride,
                   buffer.width * 4 * sizeof(float));
        }
    }
    
    return true;
}

void CPUFallback::ReleaseTexture(GPUTexture texture) {
    auto* cpu_texture = GetCPUTexture(texture);
    if (!cpu_texture) {
        return;
    }
    
    // Find and remove from allocated list
    auto it = std::find(allocated_textures_.begin(), allocated_textures_.end(), cpu_texture);
    if (it != allocated_textures_.end()) {
        allocated_textures_.erase(it);
    }
    
    if (cpu_texture->data) {
        delete[] cpu_texture->data;
    }
    delete cpu_texture;
}

GPUTexture CPUFallback::AllocateTexture(uint32_t width, uint32_t height) {
    auto* texture = new CPUTexture();
    texture->width = width;
    texture->height = height;
    texture->stride = width * 4; // RGBA
    
    texture->data = new float[texture->stride * height];
    memset(texture->data, 0, texture->stride * height * sizeof(float));
    
    allocated_textures_.push_back(texture);
    return reinterpret_cast<GPUTexture>(texture);
}

void CPUFallback::ExecuteBloom(
    GPUTexture input_texture,
    GPUTexture output_texture,
    const BloomParameters& params,
    uint32_t width,
    uint32_t height
) {
    auto* input = GetCPUTexture(input_texture);
    auto* output = GetCPUTexture(output_texture);

    if (!input || !output) {
        Logger::Error("CPUFallback: Invalid texture in ExecuteBloom");
        return;
    }

    BloomCPU(input, output, params);
}

void CPUFallback::ExecuteGlow(
    GPUTexture input_texture,
    GPUTexture output_texture,
    const GlowParameters& params,
    uint32_t width,
    uint32_t height
) {
    auto* input = GetCPUTexture(input_texture);
    auto* output = GetCPUTexture(output_texture);

    if (!input || !output) {
        Logger::Error("CPUFallback: Invalid texture in ExecuteGlow");
        return;
    }

    GlowCPU(input, output, params);
}

void CPUFallback::ExecuteHalation(
    GPUTexture input_texture,
    GPUTexture output_texture,
    const HalationParameters& params,
    uint32_t width,
    uint32_t height
) {
    auto* input = GetCPUTexture(input_texture);
    auto* output = GetCPUTexture(output_texture);

    if (!input || !output) {
        Logger::Error("CPUFallback: Invalid texture in ExecuteHalation");
        return;
    }

    HalationCPU(input, output, params);
}

void CPUFallback::ExecuteGrain(
    GPUTexture input_texture,
    GPUTexture output_texture,
    const GrainParameters& params,
    uint32_t frame_number,
    uint32_t width,
    uint32_t height
) {
    auto* input = GetCPUTexture(input_texture);
    auto* output = GetCPUTexture(output_texture);

    if (!input || !output) {
        Logger::Error("CPUFallback: Invalid texture in ExecuteGrain");
        return;
    }

    GrainCPU(input, output, params, frame_number);
}

void CPUFallback::ExecuteChromaticAberration(
    GPUTexture input_texture,
    GPUTexture output_texture,
    const ChromaticAberrationParameters& params,
    uint32_t width,
    uint32_t height
) {
    auto* input = GetCPUTexture(input_texture);
    auto* output = GetCPUTexture(output_texture);

    if (!input || !output) {
        Logger::Error("CPUFallback: Invalid texture in ExecuteChromaticAberration");
        return;
    }

    ChromaticAberrationCPU(input, output, params);
}

CPUFallback::CPUTexture* CPUFallback::GetCPUTexture(GPUTexture handle) {
    return reinterpret_cast<CPUTexture*>(handle);
}

void CPUFallback::BloomCPU(
    const CPUTexture* input,
    CPUTexture* output,
    const BloomParameters& params
) {
    // 1. Extract luminance and apply shadow lift
    auto* temp = new CPUTexture();
    temp->width = input->width;
    temp->height = input->height;
    temp->stride = input->stride;
    temp->data = new float[temp->stride * temp->height];
    
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);
            
            float r = input->data[idx + 0];
            float g = input->data[idx + 1];
            float b = input->data[idx + 2];
            float a = input->data[idx + 3];
            
            float luma = MathUtils::Luminance(r, g, b);
            float boost = powf(1.0f - luma, 2.2f) * 0.3f;
            
            temp->data[idx + 0] = r + boost;
            temp->data[idx + 1] = g + boost;
            temp->data[idx + 2] = b + boost;
            temp->data[idx + 3] = a;
        }
    }
    
    // 2. Apply Gaussian blur
    auto* blurred = new CPUTexture();
    blurred->width = temp->width;
    blurred->height = temp->height;
    blurred->stride = temp->stride;
    blurred->data = new float[blurred->stride * blurred->height];
    
    GaussianBlurCPU(temp, blurred, params.radius);
    
    // 3. Additive blend with tint
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);
            
            float base_r = input->data[idx + 0];
            float base_g = input->data[idx + 1];
            float base_b = input->data[idx + 2];
            float base_a = input->data[idx + 3];
            
            float bloom_r = blurred->data[idx + 0] * params.tint_r;
            float bloom_g = blurred->data[idx + 1] * params.tint_g;
            float bloom_b = blurred->data[idx + 2] * params.tint_b;
            
            output->data[idx + 0] = base_r + bloom_r * params.amount;
            output->data[idx + 1] = base_g + bloom_g * params.amount;
            output->data[idx + 2] = base_b + bloom_b * params.amount;
            output->data[idx + 3] = base_a;
        }
    }
    
    delete[] temp->data;
    delete temp;
    delete[] blurred->data;
    delete blurred;
}

void CPUFallback::GlowCPU(
    const CPUTexture* input,
    CPUTexture* output,
    const GlowParameters& params
) {
    // 1. Extract highlights above threshold with smooth falloff
    auto* highlights = new CPUTexture();
    highlights->width = input->width;
    highlights->height = input->height;
    highlights->stride = input->stride;
    highlights->data = new float[highlights->stride * highlights->height];

    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);

            float r = input->data[idx + 0];
            float g = input->data[idx + 1];
            float b = input->data[idx + 2];

            float luma = MathUtils::Luminance(r, g, b);

            if (luma > params.threshold) {
                float scale = (luma - params.threshold) / (1.0f - params.threshold + 0.001f);
                scale = powf(scale, 0.8f); // Softer falloff for more natural glow
                highlights->data[idx + 0] = r * scale;
                highlights->data[idx + 1] = g * scale;
                highlights->data[idx + 2] = b * scale;
            } else {
                highlights->data[idx + 0] = 0.0f;
                highlights->data[idx + 1] = 0.0f;
                highlights->data[idx + 2] = 0.0f;
            }
            highlights->data[idx + 3] = 0.0f;
        }
    }

    // 2. Blur highlights with anisotropic radius
    auto* blurred = new CPUTexture();
    blurred->width = highlights->width;
    blurred->height = highlights->height;
    blurred->stride = highlights->stride;
    blurred->data = new float[blurred->stride * blurred->height];

    // Use separate horizontal and vertical blurs for anisotropic effect
    // First horizontal blur with radius_x
    auto* temp_h = new CPUTexture();
    temp_h->width = highlights->width;
    temp_h->height = highlights->height;
    temp_h->stride = highlights->stride;
    temp_h->data = new float[temp_h->stride * temp_h->height];

    GaussianBlurCPU(highlights, temp_h, params.radius_x / 3.0f); // Horizontal

    // Then vertical blur with radius_y
    GaussianBlurCPU(temp_h, blurred, params.radius_y / 3.0f); // Vertical

    delete[] temp_h->data;
    delete temp_h;

    // 3. Apply desaturation, tint, and blend modes
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);

            float orig_r = input->data[idx + 0];
            float orig_g = input->data[idx + 1];
            float orig_b = input->data[idx + 2];
            float orig_a = input->data[idx + 3];

            float glow_r = blurred->data[idx + 0];
            float glow_g = blurred->data[idx + 1];
            float glow_b = blurred->data[idx + 2];

            // Apply desaturation to glow
            if (params.desaturation > 0.0f) {
                float glow_luma = MathUtils::Luminance(glow_r, glow_g, glow_b);
                glow_r = glow_r * (1.0f - params.desaturation) + glow_luma * params.desaturation;
                glow_g = glow_g * (1.0f - params.desaturation) + glow_luma * params.desaturation;
                glow_b = glow_b * (1.0f - params.desaturation) + glow_luma * params.desaturation;
            }

            // Apply tint to glow
            glow_r *= params.tint_r;
            glow_g *= params.tint_g;
            glow_b *= params.tint_b;

            // Apply blend mode
            float final_r, final_g, final_b;
            if (params.blend_mode == 0) { // Screen
                final_r = 1.0f - (1.0f - orig_r) * (1.0f - glow_r * params.intensity);
                final_g = 1.0f - (1.0f - orig_g) * (1.0f - glow_g * params.intensity);
                final_b = 1.0f - (1.0f - orig_b) * (1.0f - glow_b * params.intensity);
            } else if (params.blend_mode == 1) { // Add
                final_r = orig_r + glow_r * params.intensity;
                final_g = orig_g + glow_g * params.intensity;
                final_b = orig_b + glow_b * params.intensity;
            } else { // Normal (overlay-like)
                final_r = orig_r + glow_r * params.intensity;
                final_g = orig_g + glow_g * params.intensity;
                final_b = orig_b + glow_b * params.intensity;
            }

            output->data[idx + 0] = MathUtils::Clamp(final_r, 0.0f, 1.0f);
            output->data[idx + 1] = MathUtils::Clamp(final_g, 0.0f, 1.0f);
            output->data[idx + 2] = MathUtils::Clamp(final_b, 0.0f, 1.0f);
            output->data[idx + 3] = orig_a;
        }
    }

    delete[] highlights->data;
    delete highlights;
    delete[] blurred->data;
    delete blurred;
}

void CPUFallback::HalationCPU(
    const CPUTexture* input,
    CPUTexture* output,
    const HalationParameters& params
) {
    // 1. Extract extreme highlights with color control using HSL
    auto* fringe_highlights = new CPUTexture();
    fringe_highlights->width = input->width;
    fringe_highlights->height = input->height;
    fringe_highlights->stride = input->stride;
    fringe_highlights->data = new float[fringe_highlights->stride * fringe_highlights->height];

    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);

            float r = input->data[idx + 0];
            float g = input->data[idx + 1];
            float b = input->data[idx + 2];

            float luma = MathUtils::Luminance(r, g, b);

            // Only extract extreme highlights
            if (luma > params.threshold) {
                // Use HSL to RGB for precise color control
                float h = params.hue / 360.0f; // Normalize to [0,1]
                float s = params.saturation;
                float l = 0.5f; // Medium lightness for fringe

                // HSL to RGB conversion
                float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
                float x_val = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
                float m = l - c / 2.0f;

                float fringe_r, fringe_g, fringe_b;
                if (h < 1.0f/6.0f) {
                    fringe_r = c; fringe_g = x_val; fringe_b = 0.0f;
                } else if (h < 2.0f/6.0f) {
                    fringe_r = x_val; fringe_g = c; fringe_b = 0.0f;
                } else if (h < 3.0f/6.0f) {
                    fringe_r = 0.0f; fringe_g = c; fringe_b = x_val;
                } else if (h < 4.0f/6.0f) {
                    fringe_r = 0.0f; fringe_g = x_val; fringe_b = c;
                } else if (h < 5.0f/6.0f) {
                    fringe_r = x_val; fringe_g = 0.0f; fringe_b = c;
                } else {
                    fringe_r = c; fringe_g = 0.0f; fringe_b = x_val;
                }

                fringe_r += m;
                fringe_g += m;
                fringe_b += m;

                // Scale by extracted brightness with smooth falloff
                float scale = (luma - params.threshold) / (1.0f - params.threshold + 0.001f);
                scale = powf(scale, 0.7f); // Softer falloff for film-like halation
                fringe_highlights->data[idx + 0] = fringe_r * scale;
                fringe_highlights->data[idx + 1] = fringe_g * scale;
                fringe_highlights->data[idx + 2] = fringe_b * scale;
            } else {
                fringe_highlights->data[idx + 0] = 0.0f;
                fringe_highlights->data[idx + 1] = 0.0f;
                fringe_highlights->data[idx + 2] = 0.0f;
            }
            fringe_highlights->data[idx + 3] = 0.0f;
        }
    }

    // 2. Blur fringe
    auto* blurred = new CPUTexture();
    blurred->width = fringe_highlights->width;
    blurred->height = fringe_highlights->height;
    blurred->stride = fringe_highlights->stride;
    blurred->data = new float[blurred->stride * blurred->height];

    GaussianBlurCPU(fringe_highlights, blurred, params.spread);

    // 3. Additive blend with fringe
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);

            output->data[idx + 0] = input->data[idx + 0] + blurred->data[idx + 0] * params.intensity;
            output->data[idx + 1] = input->data[idx + 1] + blurred->data[idx + 1] * params.intensity;
            output->data[idx + 2] = input->data[idx + 2] + blurred->data[idx + 2] * params.intensity;
            output->data[idx + 3] = input->data[idx + 3];
        }
    }

    delete[] fringe_highlights->data;
    delete fringe_highlights;
    delete[] blurred->data;
    delete blurred;
}

void CPUFallback::GrainCPU(
    const CPUTexture* input,
    CPUTexture* output,
    const GrainParameters& params,
    uint32_t frame_number
) {
    float time_z = frame_number / 30.0f; // Temporal stability

    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);

            float r = input->data[idx + 0];
            float g = input->data[idx + 1];
            float b = input->data[idx + 2];
            float a = input->data[idx + 3];

            // Calculate luminance
            float luma = MathUtils::Luminance(r, g, b);

            // Determine grain intensity based on luminance ranges
            float grain_intensity = 0.0f;
            if (luma < 0.33f) {
                // Shadows
                grain_intensity = params.shadows_amount;
            } else if (luma < 0.66f) {
                // Midtones
                grain_intensity = params.mids_amount;
            } else {
                // Highlights
                grain_intensity = params.highlights_amount;
            }

            // Apply saturation effect
            grain_intensity *= (1.0f - params.saturation) + params.saturation * luma;

            // Generate stable noise
            float noise_x = x / params.size;
            float noise_y = y / params.size;
            float noise = MathUtils::PerlinNoise3D(noise_x, noise_y, time_z);

            // Remap noise to [-1, 1]
            noise = (noise - 0.5f) * 2.0f;

            // Apply grain with proper scaling
            float grain_amount = grain_intensity * noise * 0.05f; // Reduced intensity

            output->data[idx + 0] = MathUtils::Clamp(r + grain_amount, 0.0f, 1.0f);
            output->data[idx + 1] = MathUtils::Clamp(g + grain_amount, 0.0f, 1.0f);
            output->data[idx + 2] = MathUtils::Clamp(b + grain_amount, 0.0f, 1.0f);
            output->data[idx + 3] = a;
        }
    }
}

void CPUFallback::ChromaticAberrationCPU(
    const CPUTexture* input,
    CPUTexture* output,
    const ChromaticAberrationParameters& params
) {
    float center_x = float(input->width) * 0.5f;
    float center_y = float(input->height) * 0.5f;
    float max_distance = sqrtf(center_x * center_x + center_y * center_y);

    float angle_rad = MathUtils::DegreesToRadians(params.angle);

    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);

            // Vector from center
            float dx = float(x) - center_x;
            float dy = float(y) - center_y;

            // Distance from center (normalized)
            float distance = sqrtf(dx * dx + dy * dy) / max_distance;

            // Radial offset scaled by distance
            float offset = params.amount * distance * 2.0f;

            // Directional offset
            float offset_x = cosf(angle_rad) * offset;
            float offset_y = sinf(angle_rad) * offset;

            // Sample red channel with positive offset (bilinear)
            float r_u = float(x) + offset_x;
            float r_v = float(y) + offset_y;
            float r = SampleBilinear(input, r_u, r_v, 0); // Red channel

            // Green channel stays in place
            float g = input->data[idx + 1];

            // Sample blue channel with negative offset (bilinear)
            float b_u = float(x) - offset_x;
            float b_v = float(y) - offset_y;
            float b = SampleBilinear(input, b_u, b_v, 2); // Blue channel

            float a = input->data[idx + 3];

            output->data[idx + 0] = r;
            output->data[idx + 1] = g;
            output->data[idx + 2] = b;
            output->data[idx + 3] = a;
        }
    }
}

void CPUFallback::GaussianBlurCPU(
    const CPUTexture* input,
    CPUTexture* output,
    float radius
) {
    // Optimize for performance: limit kernel size for CPU fallback
    float sigma = radius / 3.0f;
    int kernel_size = static_cast<int>(ceilf(radius * 2.0f)) | 1; // Make odd
    kernel_size = std::min(kernel_size, 15); // Limit to 15 for performance

    std::vector<float> kernel(kernel_size);
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        float x = i - half_kernel;
        kernel[i] = MathUtils::GaussianWeight(x, sigma);
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    // Horizontal pass
    auto* temp = new CPUTexture();
    temp->width = input->width;
    temp->height = input->height;
    temp->stride = input->stride;
    temp->data = new float[temp->stride * temp->height];

    HorizontalBlurCPU(input->data, temp->data, input->width, input->height,
                     kernel.data(), kernel_size);

    // Vertical pass
    VerticalBlurCPU(temp->data, output->data, output->width, output->height,
                   kernel.data(), kernel_size);

    delete[] temp->data;
    delete temp;
}

float CPUFallback::SampleBilinear(const CPUTexture* texture, float u, float v, int channel) {
    // Clamp coordinates
    u = MathUtils::Clamp(u, 0.0f, static_cast<float>(texture->width - 1));
    v = MathUtils::Clamp(v, 0.0f, static_cast<float>(texture->height - 1));

    int x0 = static_cast<int>(floorf(u));
    int y0 = static_cast<int>(floorf(v));
    int x1 = std::min(x0 + 1, static_cast<int>(texture->width - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(texture->height - 1));

    float fx = u - static_cast<float>(x0);
    float fy = v - static_cast<float>(y0);

    // Sample four pixels
    size_t idx00 = static_cast<size_t>(y0 * texture->stride + x0 * 4) + channel;
    size_t idx10 = static_cast<size_t>(y0 * texture->stride + x1 * 4) + channel;
    size_t idx01 = static_cast<size_t>(y1 * texture->stride + x0 * 4) + channel;
    size_t idx11 = static_cast<size_t>(y1 * texture->stride + x1 * 4) + channel;

    float p00 = texture->data[idx00];
    float p10 = texture->data[idx10];
    float p01 = texture->data[idx01];
    float p11 = texture->data[idx11];

    // Bilinear interpolation
    float top = p00 * (1.0f - fx) + p10 * fx;
    float bottom = p01 * (1.0f - fx) + p11 * fx;

    return top * (1.0f - fy) + bottom * fy;
}

void CPUFallback::HorizontalBlurCPU(
    const float* input,
    float* output,
    uint32_t width,
    uint32_t height,
    const float* kernel,
    int32_t kernel_size
) {
    int half_kernel = kernel_size / 2;
    
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
            
            for (int i = -half_kernel; i <= half_kernel; i++) {
                int sample_x = MathUtils::Clamp(static_cast<int>(x) + i, 0, static_cast<int>(width - 1));
                uint32_t idx = (y * width + sample_x) * 4;
                float weight = kernel[i + half_kernel];
                
                sum_r += input[idx + 0] * weight;
                sum_g += input[idx + 1] * weight;
                sum_b += input[idx + 2] * weight;
                sum_a += input[idx + 3] * weight;
            }
            
            uint32_t out_idx = (y * width + x) * 4;
            output[out_idx + 0] = sum_r;
            output[out_idx + 1] = sum_g;
            output[out_idx + 2] = sum_b;
            output[out_idx + 3] = sum_a;
        }
    }
}

void CPUFallback::VerticalBlurCPU(
    const float* input,
    float* output,
    uint32_t width,
    uint32_t height,
    const float* kernel,
    int32_t kernel_size
) {
    int half_kernel = kernel_size / 2;
    
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
            
            for (int i = -half_kernel; i <= half_kernel; i++) {
                int sample_y = MathUtils::Clamp(static_cast<int>(y) + i, 0, static_cast<int>(height - 1));
                uint32_t idx = (sample_y * width + x) * 4;
                float weight = kernel[i + half_kernel];
                
                sum_r += input[idx + 0] * weight;
                sum_g += input[idx + 1] * weight;
                sum_b += input[idx + 2] * weight;
                sum_a += input[idx + 3] * weight;
            }
            
            uint32_t out_idx = (y * width + x) * 4;
            output[out_idx + 0] = sum_r;
            output[out_idx + 1] = sum_g;
            output[out_idx + 2] = sum_b;
            output[out_idx + 3] = sum_a;
        }
    }
}

} // namespace CinematicFX

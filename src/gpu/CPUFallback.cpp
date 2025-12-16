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
    const BloomParameters& params
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
    const GlowParameters& params
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
    const HalationParameters& params
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
    uint32_t frame_number
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
    const ChromaticAberrationParameters& params
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
    // 1. Extract highlights above threshold
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
                float factor = (luma - params.threshold) / (1.0f - params.threshold);
                highlights->data[idx + 0] = r * factor;
                highlights->data[idx + 1] = g * factor;
                highlights->data[idx + 2] = b * factor;
            } else {
                highlights->data[idx + 0] = 0.0f;
                highlights->data[idx + 1] = 0.0f;
                highlights->data[idx + 2] = 0.0f;
            }
            highlights->data[idx + 3] = 0.0f;
        }
    }
    
    // 2. Blur highlights
    auto* blurred = new CPUTexture();
    blurred->width = highlights->width;
    blurred->height = highlights->height;
    blurred->stride = highlights->stride;
    blurred->data = new float[blurred->stride * blurred->height];
    
    GaussianBlurCPU(highlights, blurred, params.diffusion_radius);
    
    // 3. Additive blend
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);
            
            output->data[idx + 0] = input->data[idx + 0] + blurred->data[idx + 0] * params.intensity;
            output->data[idx + 1] = input->data[idx + 1] + blurred->data[idx + 1] * params.intensity;
            output->data[idx + 2] = input->data[idx + 2] + blurred->data[idx + 2] * params.intensity;
            output->data[idx + 3] = input->data[idx + 3];
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
    // 1. Extract extreme highlights (red channel only)
    auto* red_highlights = new CPUTexture();
    red_highlights->width = input->width;
    red_highlights->height = input->height;
    red_highlights->stride = input->stride;
    red_highlights->data = new float[red_highlights->stride * red_highlights->height];
    
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);
            
            float luma = MathUtils::Luminance(
                input->data[idx + 0],
                input->data[idx + 1],
                input->data[idx + 2]
            );
            
            if (luma > 0.9f) {
                red_highlights->data[idx + 0] = input->data[idx + 0];
            } else {
                red_highlights->data[idx + 0] = 0.0f;
            }
            red_highlights->data[idx + 1] = 0.0f;
            red_highlights->data[idx + 2] = 0.0f;
            red_highlights->data[idx + 3] = 0.0f;
        }
    }
    
    // 2. Blur red channel
    auto* blurred = new CPUTexture();
    blurred->width = red_highlights->width;
    blurred->height = red_highlights->height;
    blurred->stride = red_highlights->stride;
    blurred->data = new float[blurred->stride * blurred->height];
    
    GaussianBlurCPU(red_highlights, blurred, params.spread);
    
    // 3. Additive blend (red fringe only)
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);
            
            output->data[idx + 0] = input->data[idx + 0] + blurred->data[idx + 0] * params.intensity;
            output->data[idx + 1] = input->data[idx + 1];
            output->data[idx + 2] = input->data[idx + 2];
            output->data[idx + 3] = input->data[idx + 3];
        }
    }
    
    delete[] red_highlights->data;
    delete red_highlights;
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
            
            // Luma mapping
            float grain_intensity = MathUtils::Lerp(
                powf(1.0f - luma, 2.0f),
                luma,
                params.roughness
            );
            
            // Generate Perlin noise
            float noise_x = x / params.size;
            float noise_y = y / params.size;
            float noise = MathUtils::PerlinNoise3D(noise_x, noise_y, time_z);
            
            // Remap noise to [-1, 1]
            noise = (noise - 0.5f) * 2.0f;
            
            // Apply grain
            float grain_amount = params.amount * grain_intensity * noise * 0.1f;
            
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
    float angle_rad = MathUtils::DegreesToRadians(params.angle);
    float offset_x = cosf(angle_rad) * params.amount;
    float offset_y = sinf(angle_rad) * params.amount;
    
    for (uint32_t y = 0; y < input->height; y++) {
        for (uint32_t x = 0; x < input->width; x++) {
            uint32_t idx = (y * input->stride + x * 4);
            
            // Sample red channel with positive offset
            int r_x = MathUtils::Clamp(static_cast<int>(x + offset_x), 0, static_cast<int>(input->width - 1));
            int r_y = MathUtils::Clamp(static_cast<int>(y + offset_y), 0, static_cast<int>(input->height - 1));
            uint32_t r_idx = (r_y * input->stride + r_x * 4);
            float r = input->data[r_idx + 0];
            
            // Green channel stays in place
            float g = input->data[idx + 1];
            
            // Sample blue channel with negative offset
            int b_x = MathUtils::Clamp(static_cast<int>(x - offset_x), 0, static_cast<int>(input->width - 1));
            int b_y = MathUtils::Clamp(static_cast<int>(y - offset_y), 0, static_cast<int>(input->height - 1));
            uint32_t b_idx = (b_y * input->stride + b_x * 4);
            float b = input->data[b_idx + 2];
            
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
    // Calculate kernel
    float sigma = radius / 3.0f;
    int kernel_size = static_cast<int>(ceilf(radius * 2.0f)) | 1; // Make odd
    kernel_size = std::min(kernel_size, 99);
    
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

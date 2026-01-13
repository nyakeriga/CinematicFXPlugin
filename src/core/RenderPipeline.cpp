/*******************************************************************************
 * CinematicFX - Render Pipeline Implementation
 * 
 * Coordinates multi-pass effect rendering
 ******************************************************************************/

#include "RenderPipeline.h"
#include "../utils/Logger.h"
#include "../utils/PerformanceTimer.h"
#include <algorithm>

namespace CinematicFX {

RenderPipeline::RenderPipeline(GPUContext* gpu_context)
    : gpu_context_(gpu_context)
    , quality_preset_(QualityPreset::STANDARD)
    , profiling_enabled_(false)
    , last_frame_time_ms_(0.0f)
    , width_(0)
    , height_(0)
    , temp_texture_1_(nullptr)
    , temp_texture_2_(nullptr)
{
    if (gpu_context_) {
        texture_manager_ = std::make_unique<TextureManager>(gpu_context_);
    }
}

RenderPipeline::~RenderPipeline() {
    CleanupTextures();
}

bool RenderPipeline::RenderFrame(
    const FrameBuffer& input,
    FrameBuffer& output,
    const EffectParameters& params,
    uint32_t frame_number
) {
    // ENHANCED: Comprehensive validation before ANY processing
    if (!gpu_context_ || !gpu_context_->GetBackend()) {
        Logger::Error("RenderPipeline: No valid GPU backend");
        return false;
    }
    
    // Validate input frame buffer
    if (!input.data || input.width <= 0 || input.height <= 0) {
        Logger::Error("RenderPipeline: Invalid input frame buffer");
        return false;
    }
    
    // Validate output frame buffer
    if (!output.data || output.width <= 0 || output.height <= 0) {
        Logger::Error("RenderPipeline: Invalid output frame buffer");
        return false;
    }
    
    // Check for dimension mismatch
    if (input.width != output.width || input.height != output.height) {
        Logger::Error("RenderPipeline: Input/output dimension mismatch");
        return false;
    }

    // Update current frame dimensions
    width_ = input.width;
    height_ = input.height;
    
    PerformanceTimer frame_timer;
    if (profiling_enabled_) {
        frame_timer.Start();
    }
    
    IGPUBackend* backend = gpu_context_->GetBackend();
    if (!backend) {
        Logger::Error("RenderPipeline: Backend is NULL");
        return false;
    }
    
    // Initialize textures on first frame or resolution change
    if (!temp_texture_1_ || !temp_texture_2_) {
        InitializeTextures(input.width, input.height);
    }
    
    // Upload input to GPU
    GPUTexture input_texture = backend->UploadTexture(input);
    if (!input_texture) {
        Logger::Error("RenderPipeline: Failed to upload input texture");
        return false;
    }
    
    // Allocate output texture
    GPUTexture output_texture = backend->AllocateTexture(output.width, output.height);
    if (!output_texture) {
        backend->ReleaseTexture(input_texture);
        Logger::Error("RenderPipeline: Failed to allocate output texture");
        return false;
    }
    
    // Current texture (ping-pong between temp textures)
    GPUTexture current_texture = input_texture;
    GPUTexture next_texture = temp_texture_1_;
    
    // Skip if no effects are active
    if (!params.HasActiveEffects()) {
        // Just copy input to output
        backend->DownloadTexture(input_texture, output);
        backend->ReleaseTexture(input_texture);
        backend->ReleaseTexture(output_texture);
        return true;
    }
    
    // Apply effect passes
    try {
        // Pass 1: Bloom
        if (params.bloom.amount > 0.0f) {
            PerformanceTimer pass_timer;
            if (profiling_enabled_) pass_timer.Start();
            
            ApplyBloomPass(current_texture, next_texture, params.bloom);
            std::swap(current_texture, next_texture);
            
            if (profiling_enabled_) {
                Logger::Debug("Bloom pass: %.2f ms", pass_timer.ElapsedMs());
            }
        }
        
        // Pass 2: Glow (Pro-Mist)
        if (params.glow.intensity > 0.0f) {
            PerformanceTimer pass_timer;
            if (profiling_enabled_) pass_timer.Start();
            
            ApplyGlowPass(current_texture, next_texture, params.glow);
            std::swap(current_texture, next_texture);
            
            if (profiling_enabled_) {
                Logger::Debug("Glow pass: %.2f ms", pass_timer.ElapsedMs());
            }
        }
        
        // Pass 3: Halation (Film Fringe)
        if (params.halation.enabled && params.halation.intensity > 0.0f) {
            PerformanceTimer pass_timer;
            if (profiling_enabled_) pass_timer.Start();

            ApplyHalationPass(current_texture, next_texture, params.halation);
            std::swap(current_texture, next_texture);

            if (profiling_enabled_) {
                Logger::Debug("Halation pass: %.2f ms", pass_timer.ElapsedMs());
            }
        }
        
        // Pass 4: Chromatic Aberration
        if (params.chromatic_aberration.enabled && params.chromatic_aberration.amount > 0.0f) {
            PerformanceTimer pass_timer;
            if (profiling_enabled_) pass_timer.Start();

            ApplyChromaticAberrationPass(current_texture, next_texture,
                                        params.chromatic_aberration);
            std::swap(current_texture, next_texture);

            if (profiling_enabled_) {
                Logger::Debug("Chromatic Aberration pass: %.2f ms", pass_timer.ElapsedMs());
            }
        }
        
        // Pass 5: Grain (final pass)
        if (params.grain.enabled && (params.grain.amount > 0.0f ||
            params.grain.shadows_amount > 0.0f ||
            params.grain.mids_amount > 0.0f ||
            params.grain.highlights_amount > 0.0f)) {
            PerformanceTimer pass_timer;
            if (profiling_enabled_) pass_timer.Start();

            ApplyGrainPass(current_texture, output_texture, params.grain, frame_number);
            current_texture = output_texture;

            if (profiling_enabled_) {
                Logger::Debug("Grain pass: %.2f ms", pass_timer.ElapsedMs());
            }
        } else {
            // No grain - copy current to output
            backend->ExecuteGrain(current_texture, output_texture, GrainParameters(), frame_number, width_, height_);
        }
        
    } catch (const std::exception& e) {
        Logger::Error("RenderPipeline: Exception during rendering: %s", e.what());
        backend->ReleaseTexture(input_texture);
        backend->ReleaseTexture(output_texture);
        return false;
    }
    
    // Download result from GPU
    backend->DownloadTexture(output_texture, output);
    
    // Synchronize GPU execution
    backend->Synchronize();
    
    // Cleanup
    backend->ReleaseTexture(input_texture);
    backend->ReleaseTexture(output_texture);
    
    // Record frame time
    if (profiling_enabled_) {
        last_frame_time_ms_ = frame_timer.ElapsedMs();
        Logger::Info("Total frame time: %.2f ms (%.1f fps)", 
                     last_frame_time_ms_, 1000.0f / last_frame_time_ms_);
        
        // Warn if slow
        if (last_frame_time_ms_ > 41.67f) { // < 24 fps
            Logger::Warning("Slow rendering detected (%.1f fps). Consider reducing effect radius or using lower quality preset.",
                          1000.0f / last_frame_time_ms_);
        }
    }
    
    return true;
}

void RenderPipeline::SetQualityPreset(QualityPreset preset) {
    quality_preset_ = preset;
    Logger::Info("Quality preset changed to: %d", static_cast<int>(preset));
}

void RenderPipeline::InitializeTextures(uint32_t width, uint32_t height) {
    if (!gpu_context_ || !gpu_context_->GetBackend()) {
        return;
    }
    
    IGPUBackend* backend = gpu_context_->GetBackend();
    
    // Cleanup old textures
    CleanupTextures();
    
    // Allocate temporary textures for ping-pong rendering
    temp_texture_1_ = backend->AllocateTexture(width, height);
    temp_texture_2_ = backend->AllocateTexture(width, height);
    
    if (!temp_texture_1_ || !temp_texture_2_) {
        Logger::Error("RenderPipeline: Failed to allocate temporary textures");
        CleanupTextures();
    }
}

void RenderPipeline::CleanupTextures() {
    if (!gpu_context_ || !gpu_context_->GetBackend()) {
        return;
    }
    
    IGPUBackend* backend = gpu_context_->GetBackend();
    
    if (temp_texture_1_) {
        backend->ReleaseTexture(temp_texture_1_);
        temp_texture_1_ = nullptr;
    }
    
    if (temp_texture_2_) {
        backend->ReleaseTexture(temp_texture_2_);
        temp_texture_2_ = nullptr;
    }
}

void RenderPipeline::ApplyBloomPass(
    GPUTexture input,
    GPUTexture output,
    const BloomParameters& params
) {
    gpu_context_->GetBackend()->ExecuteBloom(input, output, params, width_, height_);
}

void RenderPipeline::ApplyGlowPass(
    GPUTexture input,
    GPUTexture output,
    const GlowParameters& params
) {
    gpu_context_->GetBackend()->ExecuteGlow(input, output, params, width_, height_);
}

void RenderPipeline::ApplyHalationPass(
    GPUTexture input,
    GPUTexture output,
    const HalationParameters& params
) {
    gpu_context_->GetBackend()->ExecuteHalation(input, output, params, width_, height_);
}

void RenderPipeline::ApplyChromaticAberrationPass(
    GPUTexture input,
    GPUTexture output,
    const ChromaticAberrationParameters& params
) {
    gpu_context_->GetBackend()->ExecuteChromaticAberration(input, output, params, width_, height_);
}

void RenderPipeline::ApplyGrainPass(
    GPUTexture input,
    GPUTexture output,
    const GrainParameters& params,
    uint32_t frame_number
) {
    gpu_context_->GetBackend()->ExecuteGrain(input, output, params, frame_number, width_, height_);
}

} // namespace CinematicFX

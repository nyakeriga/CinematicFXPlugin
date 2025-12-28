/*******************************************************************************
 * CinematicFX - Render Pipeline
 * 
 * Master rendering coordinator - orchestrates all effect passes
 ******************************************************************************/

#pragma once

#include "CinematicFX.h"
#include "EffectParameters.h"
#include "GPUInterface.h"
#include <memory>

namespace CinematicFX {

    /**
     * @brief Main rendering pipeline
     * 
     * Coordinates the multi-pass effect rendering:
     * Input → Bloom → Glow → Halation → Chromatic Aberration → Grain → Output
     */
    class RenderPipeline {
    public:
        RenderPipeline(GPUContext* gpu_context);
        ~RenderPipeline();

        /**
         * @brief Render a frame with all effects applied
         * @param input Input frame buffer (32-bit float RGBA)
         * @param output Output frame buffer (32-bit float RGBA)
         * @param params Effect parameters
         * @param frame_number Frame number (for temporal grain stability)
         * @return true if rendering successful
         */
        bool RenderFrame(
            const FrameBuffer& input,
            FrameBuffer& output,
            const EffectParameters& params,
            uint32_t frame_number
        );

        /**
         * @brief Set quality preset (affects blur samples, etc.)
         * @param preset Quality level
         */
        void SetQualityPreset(QualityPreset preset);

        /**
         * @brief Get last frame render time in milliseconds
         * @return Render time (0 if no frame rendered yet)
         */
        float GetLastFrameTime() const { return last_frame_time_ms_; }

        /**
         * @brief Enable/disable performance profiling
         * @param enable true to enable detailed per-pass timing
         */
        void SetProfilingEnabled(bool enable) { profiling_enabled_ = enable; }

    private:
        GPUContext* gpu_context_;
        std::unique_ptr<TextureManager> texture_manager_;

        QualityPreset quality_preset_;
        bool profiling_enabled_;
        float last_frame_time_ms_;

        // Current frame dimensions
        uint32_t width_;
        uint32_t height_;

        // Intermediate textures (reused across frames)
        GPUTexture temp_texture_1_;
        GPUTexture temp_texture_2_;

        // Initialize intermediate textures (called on first frame)
        void InitializeTextures(uint32_t width, uint32_t height);

        // Cleanup textures
        void CleanupTextures();

        // Effect pass helpers
        void ApplyBloomPass(
            GPUTexture input,
            GPUTexture output,
            const BloomParameters& params
        );

        void ApplyGlowPass(
            GPUTexture input,
            GPUTexture output,
            const GlowParameters& params
        );

        void ApplyHalationPass(
            GPUTexture input,
            GPUTexture output,
            const HalationParameters& params
        );

        void ApplyChromaticAberrationPass(
            GPUTexture input,
            GPUTexture output,
            const ChromaticAberrationParameters& params
        );

        void ApplyGrainPass(
            GPUTexture input,
            GPUTexture output,
            const GrainParameters& params,
            uint32_t frame_number
        );
    };

} // namespace CinematicFX

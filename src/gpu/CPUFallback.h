/*******************************************************************************
 * CinematicFX - CPU Fallback Backend
 * 
 * Software rendering fallback (all platforms)
 ******************************************************************************/

#pragma once

#include "../include/GPUInterface.h"
#include <vector>

namespace CinematicFX {

    /**
     * @brief CPU software rendering backend (fallback)
     * 
     * Implements all effects using CPU SIMD where possible.
     * Performance is slower than GPU but guarantees compatibility.
     */
    class CPUFallback : public IGPUBackend {
    public:
        CPUFallback();
        ~CPUFallback() override;

        // IGPUBackend implementation
        bool Initialize() override;
        void Shutdown() override;
        GPUBackendType GetType() const override { return GPUBackendType::CPU; }
        const char* GetDeviceName() const override;
        uint64_t GetAvailableMemory() const override { return 0; }

        GPUTexture UploadTexture(const FrameBuffer& buffer) override;
        bool DownloadTexture(GPUTexture texture, FrameBuffer& buffer) override;
        void ReleaseTexture(GPUTexture texture) override;
        GPUTexture AllocateTexture(uint32_t width, uint32_t height) override;

        void ExecuteBloom(
            GPUTexture input_texture,
            GPUTexture output_texture,
            const BloomParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteGlow(
            GPUTexture input_texture,
            GPUTexture output_texture,
            const GlowParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteHalation(
            GPUTexture input_texture,
            GPUTexture output_texture,
            const HalationParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteGrain(
            GPUTexture input_texture,
            GPUTexture output_texture,
            const GrainParameters& params,
            uint32_t frame_number,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteChromaticAberration(
            GPUTexture input_texture,
            GPUTexture output_texture,
            const ChromaticAberrationParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void Synchronize() override { /* No-op for CPU */ }

    private:
        struct CPUTexture {
            float* data;         // RGBA float data
            uint32_t width;
            uint32_t height;
            uint32_t stride;
        };

        std::vector<CPUTexture*> allocated_textures_;
        bool initialized_;

        // Helper: Convert GPUTexture handle to CPUTexture
        CPUTexture* GetCPUTexture(GPUTexture handle);

        // CPU implementations of effects
        void BloomCPU(
            const CPUTexture* input,
            CPUTexture* output,
            const BloomParameters& params
        );

        void GlowCPU(
            const CPUTexture* input,
            CPUTexture* output,
            const GlowParameters& params
        );

        void HalationCPU(
            const CPUTexture* input,
            CPUTexture* output,
            const HalationParameters& params
        );

        void GrainCPU(
            const CPUTexture* input,
            CPUTexture* output,
            const GrainParameters& params,
            uint32_t frame_number
        );

        void ChromaticAberrationCPU(
            const CPUTexture* input,
            CPUTexture* output,
            const ChromaticAberrationParameters& params
        );

        // CPU blur implementation (separable Gaussian)
        void GaussianBlurCPU(
            const CPUTexture* input,
            CPUTexture* output,
            float radius
        );

        // CPU horizontal blur pass
        void HorizontalBlurCPU(
            const float* input,
            float* output,
            uint32_t width,
            uint32_t height,
            const float* kernel,
            int32_t kernel_size
        );

        // CPU vertical blur pass
        void VerticalBlurCPU(
            const float* input,
            float* output,
            uint32_t width,
            uint32_t height,
            const float* kernel,
            int32_t kernel_size
        );

        // Bilinear sampling helper
        float SampleBilinear(const CPUTexture* texture, float u, float v, int channel);
    };

} // namespace CinematicFX

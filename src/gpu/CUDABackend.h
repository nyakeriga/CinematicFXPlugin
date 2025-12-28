/*******************************************************************************
 * CinematicFX - CUDA Backend Implementation
 * 
 * NVIDIA CUDA GPU acceleration (Windows/Linux)
 ******************************************************************************/

#pragma once

#include "../include/GPUInterface.h"
#include <vector>

#ifdef CINEMATICFX_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace CinematicFX {

    /**
     * @brief CUDA GPU backend implementation
     */
    class CUDABackend : public IGPUBackend {
    public:
        CUDABackend();
        ~CUDABackend() override;

        // IGPUBackend implementation
        bool Initialize() override;
        void Shutdown() override;
        GPUBackendType GetType() const override { return GPUBackendType::CUDA; }
        const char* GetDeviceName() const override;
        uint64_t GetAvailableMemory() const override;

        GPUTexture UploadTexture(const FrameBuffer& buffer) override;
        bool DownloadTexture(GPUTexture texture, FrameBuffer& buffer) override;
        void ReleaseTexture(GPUTexture texture) override;
        GPUTexture AllocateTexture(uint32_t width, uint32_t height) override;

        void ExecuteBloom(
            GPUTexture input,
            GPUTexture output,
            const BloomParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteGlow(
            GPUTexture input,
            GPUTexture output,
            const GlowParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteHalation(
            GPUTexture input,
            GPUTexture output,
            const HalationParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteGrain(
            GPUTexture input,
            GPUTexture output,
            const GrainParameters& params,
            uint32_t frame_number,
            uint32_t width,
            uint32_t height
        ) override;

        void ExecuteChromaticAberration(
            GPUTexture input,
            GPUTexture output,
            const ChromaticAberrationParameters& params,
            uint32_t width,
            uint32_t height
        ) override;

        void Synchronize() override;

        /**
         * @brief Check if CUDA is available on this system
         * @return true if CUDA runtime and compatible GPU found
         */
        static bool IsAvailable();

    private:
        struct CUDAContext;
        CUDAContext* cuda_ctx_;

        char device_name_[256];
        bool initialized_;
    };

} // namespace CinematicFX

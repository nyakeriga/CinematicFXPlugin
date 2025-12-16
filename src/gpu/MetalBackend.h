/*******************************************************************************
 * CinematicFX - Metal Backend Implementation
 * 
 * Apple Metal GPU acceleration (macOS)
 ******************************************************************************/

#pragma once

#include "GPUBackend.h"
#include <vector>

// Forward declare Metal types (avoid including Objective-C in C++ header)
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace CinematicFX {

    /**
     * @brief Metal GPU backend implementation
     */
    class MetalBackend : public GPUBackendBase {
    public:
        MetalBackend();
        ~MetalBackend() override;

        // IGPUBackend implementation
        bool Initialize() override;
        void Shutdown() override;
        GPUBackendType GetType() const override { return GPUBackendType::METAL; }
        const char* GetDeviceName() const override;
        uint64_t GetAvailableMemory() const override;

        GPUTexture UploadTexture(const FrameBuffer& buffer) override;
        bool DownloadTexture(GPUTexture texture, FrameBuffer& buffer) override;
        void ReleaseTexture(GPUTexture texture) override;
        GPUTexture AllocateTexture(uint32_t width, uint32_t height) override;

        void ExecuteBloom(
            GPUTexture input,
            GPUTexture output,
            const BloomParameters& params
        ) override;

        void ExecuteGlow(
            GPUTexture input,
            GPUTexture output,
            const GlowParameters& params
        ) override;

        void ExecuteHalation(
            GPUTexture input,
            GPUTexture output,
            const HalationParameters& params
        ) override;

        void ExecuteGrain(
            GPUTexture input,
            GPUTexture output,
            const GrainParameters& params,
            uint32_t frame_number
        ) override;

        void ExecuteChromaticAberration(
            GPUTexture input,
            GPUTexture output,
            const ChromaticAberrationParameters& params
        ) override;

        void Synchronize() override;

        /**
         * @brief Check if Metal is available on this system
         * @return true if Metal framework available (macOS 10.14+)
         */
        static bool IsAvailable();

    private:
        struct MetalTexture {
            id texture;          // MTLTexture* (opaque in C++)
            uint32_t width;
            uint32_t height;
        };

        // Metal objects (Objective-C pointers, managed in .mm file)
        id device_;              // MTLDevice*
        id command_queue_;       // MTLCommandQueue*
        id library_;             // MTLLibrary*
        
        std::vector<MetalTexture*> allocated_textures_;
        char device_name_[256];
        bool initialized_;

        // Helper: Convert GPUTexture handle to MetalTexture
        MetalTexture* GetMetalTexture(GPUTexture handle);

        // Helper: Create compute pipeline state for shader
        id CreatePipelineState(const char* function_name);

        // Helper: Encode compute command
        void EncodeComputeCommand(
            id pipeline_state,
            GPUTexture input,
            GPUTexture output,
            const void* params,
            size_t params_size
        );
    };

} // namespace CinematicFX

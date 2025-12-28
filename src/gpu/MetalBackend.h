/*******************************************************************************
 * CinematicFX - Advanced Metal Backend Implementation
 * 
 * Apple Metal GPU acceleration (macOS) with full custom shader pipeline
 ******************************************************************************/

#pragma once

#include "GPUBackend.h"
#include <vector>

// Forward declare Metal types (avoid including Objective-C in C++ header)
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
typedef struct objc_object* MTLComputePipelineState;
typedef struct objc_object* MTLTexture;
#define id_MTLComputePipelineState id
#define id_MTLTexture id
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
        GPUTexture AllocateTexture(uint32_t width, uint32_t height, bool is_temporary);

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
         * @brief Check if Metal is available on this system
         * @return true if Metal framework available (macOS 10.14+)
         */
        static bool IsAvailable();

    private:
        // Forward declaration of implementation class
        struct Impl;
        Impl* impl_;

        // Metal texture wrapper with metadata
        struct MetalTexture {
            id texture;          // MTLTexture* (opaque in C++)
            uint32_t width;
            uint32_t height;
            bool is_temporary;   // For tracking temporary allocations
        };

        // Performance profiling data structure
        struct ProfilingData {
            double total_time_ms;
            uint32_t calls_count;
            std::string last_operation;
            
            ProfilingData() : total_time_ms(0.0), calls_count(0) {}
        };

        bool initialized_;

        // Helper: Convert GPUTexture handle to MetalTexture
        MetalTexture* GetMetalTexture(GPUTexture handle);

        // Helper: Create compute pipeline state for shader
        id GetPipeline(const char* function_name);

        // Helper: Encode compute command
        void EncodeComputeCommand(
            id pipeline,
            id input,
            id output,
            const void* params,
            size_t params_size,
            uint32_t width,
            uint32_t height
        );

        // Helper: Generate Gaussian kernel for blur operations
        std::vector<float> GenerateGaussianKernel(int radius);

        // Helper: Update performance profiling data
        void UpdateProfiling(const std::string& operation, double time_ms);

        // Helper: Print profiling statistics
        void PrintProfilingStats();
    };

} // namespace CinematicFX

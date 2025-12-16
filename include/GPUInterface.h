/*******************************************************************************
 * CinematicFX - GPU Abstraction Interface
 * 
 * Hardware-agnostic interface for CUDA/Metal/CPU backends
 ******************************************************************************/

#pragma once

#include "CinematicFX.h"
#include "EffectParameters.h"
#include <cstdint>
#include <memory>

namespace CinematicFX {

    /**
     * @brief GPU texture handle (opaque pointer)
     */
    using GPUTexture = void*;

    /**
     * @brief Frame buffer structure (32-bit float RGBA)
     */
    struct FrameBuffer {
        float* data;           // Pixel data (R,G,B,A interleaved)
        uint32_t width;        // Frame width in pixels
        uint32_t height;       // Frame height in pixels
        uint32_t stride;       // Row stride (usually width * 4)
        bool owns_data;        // True if this buffer owns the memory

        FrameBuffer()
            : data(nullptr), width(0), height(0), stride(0), owns_data(false) {}

        ~FrameBuffer() {
            if (owns_data && data) {
                delete[] data;
            }
        }

        // Prevent copying (use shared_ptr for shared ownership)
        FrameBuffer(const FrameBuffer&) = delete;
        FrameBuffer& operator=(const FrameBuffer&) = delete;

        // Allow moving
        FrameBuffer(FrameBuffer&& other) noexcept
            : data(other.data), width(other.width), height(other.height),
              stride(other.stride), owns_data(other.owns_data) {
            other.data = nullptr;
            other.owns_data = false;
        }
    };

    /**
     * @brief Abstract GPU backend interface
     * 
     * All GPU implementations (CUDA, Metal, CPU) must implement this interface.
     */
    class IGPUBackend {
    public:
        virtual ~IGPUBackend() = default;

        /**
         * @brief Initialize the GPU backend
         * @return true if initialization successful
         */
        virtual bool Initialize() = 0;

        /**
         * @brief Shutdown and release GPU resources
         */
        virtual void Shutdown() = 0;

        /**
         * @brief Get backend type
         * @return Backend identifier
         */
        virtual GPUBackendType GetType() const = 0;

        /**
         * @brief Get GPU device name
         * @return Device name string (e.g., "NVIDIA RTX 4090")
         */
        virtual const char* GetDeviceName() const = 0;

        /**
         * @brief Get available GPU memory in bytes
         * @return Available memory (0 if CPU backend)
         */
        virtual uint64_t GetAvailableMemory() const = 0;

        /**
         * @brief Upload frame buffer to GPU texture
         * @param buffer CPU frame buffer to upload
         * @return GPU texture handle (nullptr on failure)
         */
        virtual GPUTexture UploadTexture(const FrameBuffer& buffer) = 0;

        /**
         * @brief Download GPU texture to CPU frame buffer
         * @param texture Source GPU texture
         * @param buffer Destination CPU frame buffer
         * @return true if successful
         */
        virtual bool DownloadTexture(GPUTexture texture, FrameBuffer& buffer) = 0;

        /**
         * @brief Release GPU texture
         * @param texture GPU texture handle
         */
        virtual void ReleaseTexture(GPUTexture texture) = 0;

        /**
         * @brief Allocate temporary GPU texture (for intermediate passes)
         * @param width Texture width
         * @param height Texture height
         * @return GPU texture handle
         */
        virtual GPUTexture AllocateTexture(uint32_t width, uint32_t height) = 0;

        /**
         * @brief Execute Bloom effect pass
         * @param input Input GPU texture
         * @param output Output GPU texture
         * @param params Bloom parameters
         */
        virtual void ExecuteBloom(
            GPUTexture input,
            GPUTexture output,
            const BloomParameters& params
        ) = 0;

        /**
         * @brief Execute Glow effect pass
         * @param input Input GPU texture
         * @param output Output GPU texture
         * @param params Glow parameters
         */
        virtual void ExecuteGlow(
            GPUTexture input,
            GPUTexture output,
            const GlowParameters& params
        ) = 0;

        /**
         * @brief Execute Halation effect pass
         * @param input Input GPU texture
         * @param output Output GPU texture
         * @param params Halation parameters
         */
        virtual void ExecuteHalation(
            GPUTexture input,
            GPUTexture output,
            const HalationParameters& params
        ) = 0;

        /**
         * @brief Execute Grain effect pass
         * @param input Input GPU texture
         * @param output Output GPU texture
         * @param params Grain parameters
         * @param frame_number Frame number for temporal stability
         */
        virtual void ExecuteGrain(
            GPUTexture input,
            GPUTexture output,
            const GrainParameters& params,
            uint32_t frame_number
        ) = 0;

        /**
         * @brief Execute Chromatic Aberration effect pass
         * @param input Input GPU texture
         * @param output Output GPU texture
         * @param params Chromatic aberration parameters
         */
        virtual void ExecuteChromaticAberration(
            GPUTexture input,
            GPUTexture output,
            const ChromaticAberrationParameters& params
        ) = 0;

        /**
         * @brief Synchronize GPU execution (wait for all operations to complete)
         */
        virtual void Synchronize() = 0;
    };

    /**
     * @brief GPU context manager
     * 
     * Handles automatic backend selection and fallback.
     */
    class GPUContext {
    public:
        /**
         * @brief Create GPU context with automatic backend selection
         * @param preferred_backend Preferred backend type
         * @return Unique pointer to GPU context
         */
        static std::unique_ptr<GPUContext> Create(
            GPUBackendType preferred_backend = GPUBackendType::CUDA
        );

        /**
         * @brief Destructor
         */
        ~GPUContext();

        /**
         * @brief Get active backend
         * @return Pointer to active backend (never null)
         */
        IGPUBackend* GetBackend() const { return backend_.get(); }

        /**
         * @brief Get active backend type
         * @return Backend type
         */
        GPUBackendType GetBackendType() const;

        /**
         * @brief Check if GPU acceleration is active
         * @return true if using CUDA or Metal, false if CPU fallback
         */
        bool IsGPUAccelerated() const;

        /**
         * @brief Force fallback to CPU backend
         * (Used when GPU operations fail)
         */
        void FallbackToCPU();

    private:
        GPUContext() = default;

        std::unique_ptr<IGPUBackend> backend_;

        // Detect best available backend
        static GPUBackendType DetectBestBackend();

        // Create backend instance
        static std::unique_ptr<IGPUBackend> CreateBackend(GPUBackendType type);
    };

    /**
     * @brief Texture manager for efficient GPU memory reuse
     */
    class TextureManager {
    public:
        TextureManager(GPUContext* context);
        ~TextureManager();

        /**
         * @brief Acquire a temporary texture (from pool or allocate new)
         * @param width Texture width
         * @param height Texture height
         * @return GPU texture handle
         */
        GPUTexture AcquireTexture(uint32_t width, uint32_t height);

        /**
         * @brief Release temporary texture back to pool
         * @param texture GPU texture handle
         */
        void ReleaseTexture(GPUTexture texture);

        /**
         * @brief Clear all cached textures
         */
        void ClearPool();

    private:
        GPUContext* context_;
        struct TexturePool;
        std::unique_ptr<TexturePool> pool_;
    };

} // namespace CinematicFX

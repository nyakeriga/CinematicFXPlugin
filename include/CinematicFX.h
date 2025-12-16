/*******************************************************************************
 * CinematicFX - Professional Cinematic Effects Plugin for Adobe Premiere Pro
 * 
 * Copyright (c) 2025 Pol Casals
 * Licensed under Commercial License
 * 
 * Main Plugin Header - Public API
 ******************************************************************************/

#pragma once

#define CINEMATICFX_VERSION_MAJOR 1
#define CINEMATICFX_VERSION_MINOR 0
#define CINEMATICFX_VERSION_PATCH 0

// Platform detection
#if defined(_WIN32) && !defined(CINEMATICFX_PLATFORM_WINDOWS)
    #define CINEMATICFX_PLATFORM_WINDOWS
#endif

#if defined(_WIN32)
    #ifdef CINEMATICFX_EXPORTS
        #define CINEMATICFX_API __declspec(dllexport)
    #else
        #define CINEMATICFX_API __declspec(dllimport)
    #endif
#elif defined(__APPLE__)
    #define CINEMATICFX_PLATFORM_MACOS
    #define CINEMATICFX_API __attribute__((visibility("default")))
#else
    #define CINEMATICFX_API
#endif

// GPU Backend availability
#ifdef __CUDACC__
    #define CINEMATICFX_CUDA_AVAILABLE
#endif

#ifdef __METAL_VERSION__
    #define CINEMATICFX_METAL_AVAILABLE
#endif

#include <cstdint>
#include <memory>
#include <string>

namespace CinematicFX {

    // Forward declarations
    class RenderPipeline;
    class GPUContext;
    struct EffectParameters;

    /**
     * @brief Plugin initialization result codes
     */
    enum class InitResult : int32_t {
        SUCCESS = 0,
        ERROR_SDK_VERSION_MISMATCH = -1,
        ERROR_GPU_UNAVAILABLE = -2,
        ERROR_LICENSE_INVALID = -3,
        ERROR_MEMORY_ALLOCATION = -4,
        ERROR_UNKNOWN = -999
    };

    /**
     * @brief GPU backend types (ordered by preference)
     */
    enum class GPUBackendType : uint8_t {
        CUDA = 0,      // NVIDIA CUDA (Windows/Linux)
        METAL = 1,     // Apple Metal (macOS)
        CPU = 2        // Software fallback (all platforms)
    };

    /**
     * @brief Pixel format for rendering pipeline
     */
    enum class PixelFormat : uint8_t {
        RGBA_32F,      // 32-bit float per channel (default)
        RGBA_16F,      // 16-bit float per channel (half precision)
        RGBA_8U        // 8-bit unsigned per channel (legacy)
    };

    /**
     * @brief Effect quality presets
     */
    enum class QualityPreset : uint8_t {
        DRAFT,         // Fast preview (reduced blur samples)
        STANDARD,      // Balanced quality/performance
        HIGH,          // High quality (production)
        MAXIMUM        // Maximum quality (final render)
    };

    /**
     * @brief Plugin capabilities structure
     */
    struct PluginCapabilities {
        bool cuda_available;
        bool metal_available;
        bool cpu_fallback_available;
        uint32_t max_texture_size;
        uint64_t available_gpu_memory;
        const char* gpu_device_name;
        uint32_t compute_units;
    };

    /**
     * @brief Main plugin interface
     * 
     * This is the primary entry point for the plugin.
     * Handles initialization, rendering, and cleanup.
     */
    class CINEMATICFX_API Plugin {
    public:
        /**
         * @brief Initialize the plugin
         * @param sdk_version AE SDK version number
         * @param preferred_backend Preferred GPU backend (or AUTO)
         * @return Initialization result code
         */
        static InitResult Initialize(
            uint32_t sdk_version,
            GPUBackendType preferred_backend = GPUBackendType::CUDA
        );

        /**
         * @brief Shutdown the plugin and release all resources
         */
        static void Shutdown();

        /**
         * @brief Get plugin version string
         * @return Version string (e.g., "1.0.0")
         */
        static const char* GetVersion();

        /**
         * @brief Get current GPU backend in use
         * @return Active backend type
         */
        static GPUBackendType GetActiveBackend();

        /**
         * @brief Query plugin capabilities
         * @return Capabilities structure
         */
        static PluginCapabilities GetCapabilities();

        /**
         * @brief Check if license is valid
         * @return true if licensed, false if trial/invalid
         */
        static bool IsLicenseValid();

        /**
         * @brief Activate license with key
         * @param license_key License activation key
         * @return true if activation successful
         */
        static bool ActivateLicense(const char* license_key);
    };

} // namespace CinematicFX

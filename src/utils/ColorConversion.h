/********************************************************************************
 * CinematicFX - Color Format Conversion Utilities
 *
 * Safe conversion between Adobe PF_Pixel8 (8-bit) and engine float format (32-bit)
 * Prevents direct memory access violations and color corruption
 *******************************************************************************/

#pragma once

#include <cstdint>
#include <vector>
#include <memory>

namespace CinematicFX {

    // Adobe After Effects pixel format (8-bit RGBA)
    struct PF_Pixel8 {
        uint8_t red;
        uint8_t green;
        uint8_t blue;
        uint8_t alpha;
    };

    /**
     * @brief Color conversion utilities
     *
     * Provides safe conversion between Adobe's 8-bit format and engine's 32-bit float format.
     * Never lets Adobe memory touch the engine directly.
     */
    class ColorConversion {
    public:
        /**
         * @brief Convert Adobe PF_Pixel8 buffer to engine float buffer
         * @param adobe_pixels Source Adobe pixel buffer (8-bit RGBA)
         * @param width Frame width in pixels
         * @param height Frame height in pixels
         * @param adobe_stride Adobe buffer stride (bytes per row)
         * @return Unique pointer to converted float buffer (32-bit RGBA, 0.0-1.0)
         */
        static std::unique_ptr<float[]> AdobeToEngine(
            const PF_Pixel8* adobe_pixels,
            uint32_t width,
            uint32_t height,
            uint32_t adobe_stride
        );

        /**
         * @brief Convert engine float buffer back to Adobe PF_Pixel8 buffer
         * @param engine_pixels Source engine float buffer (32-bit RGBA, 0.0-1.0)
         * @param adobe_pixels Destination Adobe pixel buffer (8-bit RGBA)
         * @param width Frame width in pixels
         * @param height Frame height in pixels
         * @param adobe_stride Adobe buffer stride (bytes per row)
         */
        static void EngineToAdobe(
            const float* engine_pixels,
            PF_Pixel8* adobe_pixels,
            uint32_t width,
            uint32_t height,
            uint32_t adobe_stride
        );

        /**
         * @brief Create intermediate buffer for engine processing
         * @param width Frame width
         * @param height Frame height
         * @return Unique pointer to float buffer (4 floats per pixel: RGBA)
         */
        static std::unique_ptr<float[]> CreateEngineBuffer(uint32_t width, uint32_t height);

        /**
         * @brief Get buffer size in floats for RGBA frame
         * @param width Frame width
         * @param height Frame height
         * @return Total float count (width * height * 4)
         */
        static size_t GetEngineBufferSize(uint32_t width, uint32_t height);

    private:
        // Prevent instantiation - static utility class
        ColorConversion() = delete;
        ~ColorConversion() = delete;

        /**
         * @brief Convert single 8-bit value to float (0-255 → 0.0-1.0)
         */
        static inline float ByteToFloat(uint8_t value) {
            return static_cast<float>(value) / 255.0f;
        }

        /**
         * @brief Convert single float to 8-bit value (0.0-1.0 → 0-255)
         */
        static inline uint8_t FloatToByte(float value) {
            // Clamp to valid range and convert
            if (value <= 0.0f) return 0;
            if (value >= 1.0f) return 255;
            return static_cast<uint8_t>(value * 255.0f + 0.5f); // Round to nearest
        }
    };

} // namespace CinematicFX
/********************************************************************************
 * CinematicFX - Color Format Conversion Implementation
 *
 * Safe conversion between Adobe PF_Pixel8 and engine float format
 *******************************************************************************/

#include "ColorConversion.h"
#include <cstring>
#include <algorithm>

namespace CinematicFX {

    std::unique_ptr<float[]> ColorConversion::AdobeToEngine(
        const PF_Pixel8* adobe_pixels,
        uint32_t width,
        uint32_t height,
        uint32_t adobe_stride
    ) {
        // Create engine buffer
        auto engine_buffer = CreateEngineBuffer(width, height);
        float* engine_pixels = engine_buffer.get();

        // Convert each pixel
        for (uint32_t y = 0; y < height; ++y) {
            // Calculate row pointers
            const PF_Pixel8* adobe_row = reinterpret_cast<const PF_Pixel8*>(
                reinterpret_cast<const char*>(adobe_pixels) + y * adobe_stride
            );
            float* engine_row = engine_pixels + y * width * 4;

            for (uint32_t x = 0; x < width; ++x) {
                const PF_Pixel8& adobe_pixel = adobe_row[x];
                float* engine_pixel = engine_row + x * 4;

                // Convert 8-bit to float (0-255 → 0.0-1.0)
                engine_pixel[0] = ByteToFloat(adobe_pixel.red);   // R
                engine_pixel[1] = ByteToFloat(adobe_pixel.green); // G
                engine_pixel[2] = ByteToFloat(adobe_pixel.blue);  // B
                engine_pixel[3] = ByteToFloat(adobe_pixel.alpha); // A
            }
        }

        return engine_buffer;
    }

    void ColorConversion::EngineToAdobe(
        const float* engine_pixels,
        PF_Pixel8* adobe_pixels,
        uint32_t width,
        uint32_t height,
        uint32_t adobe_stride
    ) {
        // Convert each pixel back
        for (uint32_t y = 0; y < height; ++y) {
            // Calculate row pointers
            PF_Pixel8* adobe_row = reinterpret_cast<PF_Pixel8*>(
                reinterpret_cast<char*>(adobe_pixels) + y * adobe_stride
            );
            const float* engine_row = engine_pixels + y * width * 4;

            for (uint32_t x = 0; x < width; ++x) {
                PF_Pixel8& adobe_pixel = adobe_row[x];
                const float* engine_pixel = engine_row + x * 4;

                // Convert float to 8-bit (0.0-1.0 → 0-255)
                adobe_pixel.red   = FloatToByte(engine_pixel[0]); // R
                adobe_pixel.green = FloatToByte(engine_pixel[1]); // G
                adobe_pixel.blue  = FloatToByte(engine_pixel[2]); // B
                adobe_pixel.alpha = FloatToByte(engine_pixel[3]); // A
            }
        }
    }

    std::unique_ptr<float[]> ColorConversion::CreateEngineBuffer(uint32_t width, uint32_t height) {
        size_t buffer_size = GetEngineBufferSize(width, height);
        return std::unique_ptr<float[]>(new float[buffer_size]);
    }

    size_t ColorConversion::GetEngineBufferSize(uint32_t width, uint32_t height) {
        return static_cast<size_t>(width) * height * 4; // RGBA floats per pixel
    }

} // namespace CinematicFX
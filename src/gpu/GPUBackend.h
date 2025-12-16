/*******************************************************************************
 * CinematicFX - GPU Backend Abstract Base Class
 * 
 * Base implementation for all GPU backends
 ******************************************************************************/

#pragma once

#include "GPUInterface.h"

namespace CinematicFX {

    /**
     * @brief Base class for GPU backends with common utilities
     */
    class GPUBackendBase : public IGPUBackend {
    public:
        virtual ~GPUBackendBase() = default;

    protected:
        /**
         * @brief Calculate Gaussian blur kernel weights
         * @param sigma Blur standard deviation
         * @param kernel_size Output kernel size (must be odd)
         * @param weights Output array of weights
         */
        static void CalculateGaussianKernel(
            float sigma,
            int32_t& kernel_size,
            float* weights
        );

        /**
         * @brief Calculate luminance from RGB
         * @param r Red component
         * @param g Green component
         * @param b Blue component
         * @return Luminance value (Rec. 709)
         */
        static inline float CalculateLuminance(float r, float g, float b) {
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        /**
         * @brief Clamp value to range [0, 1]
         */
        static inline float Clamp01(float value) {
            return (value < 0.0f) ? 0.0f : (value > 1.0f) ? 1.0f : value;
        }

        /**
         * @brief Linear interpolation
         */
        static inline float Lerp(float a, float b, float t) {
            return a + (b - a) * t;
        }
    };

} // namespace CinematicFX

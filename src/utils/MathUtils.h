/*******************************************************************************
 * CinematicFX - Math Utilities
 ******************************************************************************/

#pragma once

#include <cmath>
#include <algorithm>

namespace CinematicFX {

    /**
     * @brief Math utility functions
     */
    class MathUtils {
    public:
        /**
         * @brief Clamp value to range [min, max]
         */
        template<typename T>
        static inline T Clamp(T value, T min_val, T max_val) {
            return std::max(min_val, std::min(max_val, value));
        }
        
        /**
         * @brief Linear interpolation
         */
        static inline float Lerp(float a, float b, float t) {
            return a + (b - a) * t;
        }
        
        /**
         * @brief Smoothstep interpolation
         */
        static inline float Smoothstep(float edge0, float edge1, float x) {
            float t = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }
        
        /**
         * @brief Calculate Gaussian weight
         */
        static inline float GaussianWeight(float x, float sigma) {
            float norm = 1.0f / (sigma * sqrtf(2.0f * 3.14159265359f));
            return norm * expf(-(x * x) / (2.0f * sigma * sigma));
        }
        
        /**
         * @brief Calculate luminance from RGB (Rec. 709)
         */
        static inline float Luminance(float r, float g, float b) {
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
        
        /**
         * @brief Convert degrees to radians
         */
        static inline float DegreesToRadians(float degrees) {
            return degrees * 3.14159265359f / 180.0f;
        }
        
        /**
         * @brief Convert radians to degrees
         */
        static inline float RadiansToDegrees(float radians) {
            return radians * 180.0f / 3.14159265359f;
        }
        
        /**
         * @brief Perlin noise (1D)
         */
        static float PerlinNoise1D(float x);
        
        /**
         * @brief Perlin noise (2D)
         */
        static float PerlinNoise2D(float x, float y);
        
        /**
         * @brief Perlin noise (3D) - for temporal grain stability
         */
        static float PerlinNoise3D(float x, float y, float z);
        
    private:
        // Permutation table for Perlin noise
        static const int PERMUTATION_TABLE[512];
        
        static inline float Fade(float t) {
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        }
        
        static inline float Grad(int hash, float x, float y, float z) {
            int h = hash & 15;
            float u = h < 8 ? x : y;
            float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }
    };

} // namespace CinematicFX

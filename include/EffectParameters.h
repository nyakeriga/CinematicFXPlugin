/*******************************************************************************
 * CinematicFX - Effect Parameters Structure
 * 
 * All user-controllable parameters with keyframe support
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <cmath>

namespace CinematicFX {

    /**
     * @brief Color structure (32-bit float RGB)
     */
    struct Color {
        float r, g, b;

        Color() : r(1.0f), g(1.0f), b(1.0f) {}
        Color(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

        // Clamp to valid range [0, 1]
        void Clamp() {
            r = std::fmax(0.0f, std::fmin(1.0f, r));
            g = std::fmax(0.0f, std::fmin(1.0f, g));
            b = std::fmax(0.0f, std::fmin(1.0f, b));
        }
    };

    /**
     * @brief Bloom effect parameters
     * 
     * Creates soft atmospheric diffusion across the entire image,
     * lifting shadows and midtones for cinematic feel.
     */
    struct BloomParameters {
        float amount;          // Intensity: 0.0 = off, 1.0 = full strength
        float radius;          // Diffusion radius in pixels: 1.0 - 100.0
        float threshold;       // Luminance threshold: 0.0 - 1.0
        float shadow_lift;     // Shadow boost: 0.0 - 1.0
        float tint_r;          // Red tint: 0.0 - 1.0
        float tint_g;          // Green tint: 0.0 - 1.0
        float tint_b;          // Blue tint: 0.0 - 1.0

        BloomParameters()
            : amount(0.3f), radius(30.0f), threshold(0.0f), shadow_lift(0.5f),
              tint_r(1.0f), tint_g(1.0f), tint_b(1.0f) {}

        void Validate() {
            amount = std::fmax(0.0f, std::fmin(1.0f, amount));
            radius = std::fmax(1.0f, std::fmin(100.0f, radius));
            threshold = std::fmax(0.0f, std::fmin(1.0f, threshold));
            shadow_lift = std::fmax(0.0f, std::fmin(1.0f, shadow_lift));
            tint_r = std::fmax(0.0f, std::fmin(1.0f, tint_r));
            tint_g = std::fmax(0.0f, std::fmin(1.0f, tint_g));
            tint_b = std::fmax(0.0f, std::fmin(1.0f, tint_b));
        }
    };

    /**
     * @brief Glow (Pro-Mist Diffusion) parameters
     * 
     * Selective diffusion applied only to bright highlights,
     * mimicking classic film diffusion filters.
     */
    struct GlowParameters {
        float threshold;       // Luminance threshold: 0.0 - 1.0 (0.7 = only bright areas)
        float diffusion_radius; // Glow radius in pixels: 1.0 - 100.0
        float intensity;       // Glow strength: 0.0 - 2.0

        GlowParameters()
            : threshold(0.7f), diffusion_radius(40.0f), intensity(0.5f) {}

        void Validate() {
            threshold = std::fmax(0.0f, std::fmin(1.0f, threshold));
            diffusion_radius = std::fmax(1.0f, std::fmin(100.0f, diffusion_radius));
            intensity = std::fmax(0.0f, std::fmin(2.0f, intensity));
        }
    };

    /**
     * @brief Halation (Film Fringe) parameters
     * 
     * Red fringe effect on bright highlights, replicating
     * film stock light scattering.
     */
    struct HalationParameters {
        float intensity;       // Fringe strength: 0.0 - 1.0
        float spread;          // Fringe spread in pixels: 1.0 - 50.0

        HalationParameters()
            : intensity(0.4f), spread(15.0f) {}

        void Validate() {
            intensity = std::fmax(0.0f, std::fmin(1.0f, intensity));
            spread = std::fmax(1.0f, std::fmin(50.0f, spread));
        }
    };

    /**
     * @brief Curated Grain parameters
     * 
     * Procedural film grain with luminosity-based intensity mapping.
     * Not random per-frame; stable, cinematic grain.
     */
    struct GrainParameters {
        float amount;          // Overall grain visibility: 0.0 - 1.0
        float size;            // Grain texture scale: 0.5 - 5.0
        float roughness;       // Grain distribution: 0.0 = smooth, 1.0 = rough

        GrainParameters()
            : amount(0.2f), size(1.0f), roughness(0.5f) {}

        void Validate() {
            amount = std::fmax(0.0f, std::fmin(1.0f, amount));
            size = std::fmax(0.5f, std::fmin(5.0f, size));
            roughness = std::fmax(0.0f, std::fmin(1.0f, roughness));
        }
    };

    /**
     * @brief Chromatic Aberration parameters
     * 
     * RGB channel spatial offset to simulate lens distortion.
     */
    struct ChromaticAberrationParameters {
        float amount;          // Offset amount in pixels: 0.0 - 10.0
        float angle;           // Offset direction in degrees: 0.0 - 360.0

        ChromaticAberrationParameters()
            : amount(0.0f), angle(0.0f) {}

        void Validate() {
            amount = std::fmax(0.0f, std::fmin(10.0f, amount));
            angle = std::fmod(angle, 360.0f);
            if (angle < 0.0f) angle += 360.0f;
        }
    };

    /**
     * @brief Master effect parameters structure
     * 
     * Contains all user-controllable parameters for the entire plugin.
     * All parameters are keyframeable in Premiere Pro timeline.
     */
    struct EffectParameters {
        // Effect modules
        BloomParameters bloom;
        GlowParameters glow;
        HalationParameters halation;
        GrainParameters grain;
        ChromaticAberrationParameters chromatic_aberration;

        // Master controls
        bool output_enabled;   // Global on/off switch

        EffectParameters()
            : output_enabled(true) {}

        /**
         * @brief Validate all parameters (clamp to valid ranges)
         */
        void ValidateAll() {
            bloom.Validate();
            glow.Validate();
            halation.Validate();
            grain.Validate();
            chromatic_aberration.Validate();
        }

        /**
         * @brief Check if any effect is active
         * @return true if at least one effect has non-zero strength
         */
        bool HasActiveEffects() const {
            if (!output_enabled) return false;
            return (bloom.amount > 0.0f) ||
                   (glow.intensity > 0.0f) ||
                   (halation.intensity > 0.0f) ||
                   (grain.amount > 0.0f) ||
                   (chromatic_aberration.amount > 0.0f);
        }
    };

    /**
     * @brief Preset structure for saving/loading effect configurations
     */
    struct EffectPreset {
        char name[64];
        EffectParameters parameters;

        EffectPreset() {
            name[0] = '\0';
        }
    };

    // Factory presets
    namespace Presets {
        inline EffectParameters CinematicGlow() {
            EffectParameters params;
            params.bloom.amount = 0.4f;
            params.bloom.radius = 40.0f;
            params.glow.threshold = 0.75f;
            params.glow.diffusion_radius = 50.0f;
            params.glow.intensity = 0.6f;
            params.halation.intensity = 0.3f;
            params.grain.amount = 0.15f;
            return params;
        }

        inline EffectParameters VintageFilm() {
            EffectParameters params;
            params.bloom.amount = 0.3f;
            params.bloom.tint_r = 1.0f;
            params.bloom.tint_g = 0.95f;
            params.bloom.tint_b = 0.85f;
            params.halation.intensity = 0.6f;
            params.halation.spread = 20.0f;
            params.grain.amount = 0.35f;
            params.grain.size = 1.5f;
            params.chromatic_aberration.amount = 2.0f;
            return params;
        }

        inline EffectParameters DreamyDiffusion() {
            EffectParameters params;
            params.bloom.amount = 0.6f;
            params.bloom.radius = 60.0f;
            params.glow.threshold = 0.6f;
            params.glow.diffusion_radius = 80.0f;
            params.glow.intensity = 0.8f;
            params.grain.amount = 0.1f;
            return params;
        }

        inline EffectParameters SubtleGrain() {
            EffectParameters params;
            params.grain.amount = 0.25f;
            params.grain.size = 0.8f;
            params.grain.roughness = 0.3f;
            return params;
        }
    }

} // namespace CinematicFX

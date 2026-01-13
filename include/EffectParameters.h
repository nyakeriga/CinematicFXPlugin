/*******************************************************************************
 * CinematicFX - Effect Parameters Structure
 * 
 * All user-controllable parameters with keyframe support
 *******************************************************************************/

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
        float threshold;           // Luminance threshold: 0.0 - 1.0 (0.7 = only bright areas)
        float radius_x;            // Horizontal glow radius: 1.0 - 100.0
        float radius_y;            // Vertical glow radius: 1.0 - 100.0
        float diffusion_radius;    // Average radius for backward compatibility
        float intensity;           // Glow strength: 0.0 - 2.0
        float desaturation;        // Color desaturation: 0.0 - 1.0
        int blend_mode;            // Blend mode: 0=Screen, 1=Add, 2=Normal
        float tint_r;              // Tint red component: 0.0 - 1.0
        float tint_g;              // Tint green component: 0.0 - 1.0
        float tint_b;              // Tint blue component: 0.0 - 1.0

        GlowParameters()
            : threshold(0.7f), radius_x(20.0f), radius_y(20.0f), diffusion_radius(20.0f),
              intensity(0.5f), desaturation(0.3f), blend_mode(1),
              tint_r(1.0f), tint_g(1.0f), tint_b(1.0f) {}

        void Validate() {
            threshold = std::fmax(0.0f, std::fmin(1.0f, threshold));
            radius_x = std::fmax(1.0f, std::fmin(100.0f, radius_x));
            radius_y = std::fmax(1.0f, std::fmin(100.0f, radius_y));
            diffusion_radius = (radius_x + radius_y) * 0.5f;
            intensity = std::fmax(0.0f, std::fmin(2.0f, intensity));
            desaturation = std::fmax(0.0f, std::fmin(1.0f, desaturation));
            blend_mode = std::fmax(0, std::min(2, blend_mode));
            tint_r = std::fmax(0.0f, std::fmin(1.0f, tint_r));
            tint_g = std::fmax(0.0f, std::fmin(1.0f, tint_g));
            tint_b = std::fmax(0.0f, std::fmin(1.0f, tint_b));
        }
    };

    /**
     * @brief Halation (Film Fringe) parameters
     *
     * Red fringe effect on bright highlights, replicating
     * film stock light scattering.
     */
    struct HalationParameters {
        bool enabled;          // Enable/disable halation
        float intensity;       // Fringe strength: 0.0 - 1.0
        float spread;          // Fringe spread in pixels: 1.0 - 50.0
        float hue;             // Fringe color hue: 0.0 - 360.0
        float saturation;      // Fringe color saturation: 0.0 - 2.0
        float threshold;       // Luminance threshold: 0.0 - 1.0

        HalationParameters()
            : enabled(true), intensity(0.4f), spread(15.0f), hue(0.0f), saturation(1.0f), threshold(0.3f) {}

        void Validate() {
            intensity = std::fmax(0.0f, std::fmin(1.0f, intensity));
            spread = std::fmax(1.0f, std::fmin(50.0f, spread));
            hue = std::fmod(hue, 360.0f);
            if (hue < 0.0f) hue += 360.0f;
            saturation = std::fmax(0.0f, std::fmin(2.0f, saturation));
            threshold = std::fmax(0.0f, std::fmin(1.0f, threshold));
        }
    };

    /**
     * @brief Curated Grain parameters
     *
     * Procedural film grain with luminosity-based intensity mapping.
     * Not random per-frame; stable, cinematic grain.
     */
    struct GrainParameters {
        bool enabled;             // Enable/disable grain
        float shadows_amount;      // Shadows grain intensity: 0.0 - 1.0
        float mids_amount;         // Midtones grain intensity: 0.0 - 1.0
        float highlights_amount;   // Highlights grain intensity: 0.0 - 1.0
        float size;                // Grain texture scale: 0.5 - 5.0
        float roughness;           // Grain distribution: 0.0 = smooth, 1.0 = rough
        float saturation;          // Grain color saturation: 0.0 - 2.0
        float amount;              // Overall grain visibility (legacy): 0.0 - 1.0

        GrainParameters()
            : enabled(true), shadows_amount(0.2f), mids_amount(0.35f), highlights_amount(0.15f),
              size(1.0f), roughness(0.5f), saturation(1.0f), amount(0.2f) {}

        void Validate() {
            shadows_amount = std::fmax(0.0f, std::fmin(1.0f, shadows_amount));
            mids_amount = std::fmax(0.0f, std::fmin(1.0f, mids_amount));
            highlights_amount = std::fmax(0.0f, std::fmin(1.0f, highlights_amount));
            size = std::fmax(0.5f, std::fmin(5.0f, size));
            roughness = std::fmax(0.0f, std::fmin(1.0f, roughness));
            saturation = std::fmax(0.0f, std::fmin(2.0f, saturation));
            amount = std::fmax(0.0f, std::fmin(1.0f, amount));
        }
    };

    /**
     * @brief Chromatic Aberration parameters
     *
     * RGB channel spatial offset to simulate lens distortion.
     */
    struct ChromaticAberrationParameters {
        bool enabled;          // Enable/disable chromatic aberration
        float amount;          // Overall aberration intensity: 0.0 - 1.0
        float red_scale;       // Red channel scale: 0.5 - 2.0
        float green_scale;     // Green channel scale: 0.5 - 2.0
        float blue_scale;      // Blue channel scale: 0.5 - 2.0
        float blurriness;      // Edge softness: 0.0 - 10.0
        float angle;           // Offset direction in degrees: 0.0 - 360.0

        ChromaticAberrationParameters()
            : enabled(true), amount(0.0f), red_scale(1.0f), green_scale(1.0f), blue_scale(1.0f),
              blurriness(0.0f), angle(0.0f) {}

        void Validate() {
            amount = std::fmax(0.0f, std::fmin(1.0f, amount));
            red_scale = std::fmax(0.5f, std::fmin(2.0f, red_scale));
            green_scale = std::fmax(0.5f, std::fmin(2.0f, green_scale));
            blue_scale = std::fmax(0.5f, std::fmin(2.0f, blue_scale));
            blurriness = std::fmax(0.0f, std::fmin(10.0f, blurriness));
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
         * @return true if at least one effect has non-zero strength and is enabled
         */
        bool HasActiveEffects() const {
            if (!output_enabled) return false;
            return (bloom.amount > 0.0f) ||
                   (glow.intensity > 0.0f) ||
                   (halation.enabled && halation.intensity > 0.0f) ||
                   (grain.enabled && (grain.amount > 0.0f ||
                    grain.shadows_amount > 0.0f ||
                    grain.mids_amount > 0.0f ||
                    grain.highlights_amount > 0.0f)) ||
                   (chromatic_aberration.enabled && chromatic_aberration.amount > 0.0f);
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
            params.glow.radius_x = 50.0f;
            params.glow.radius_y = 50.0f;
            params.glow.diffusion_radius = 50.0f;
            params.glow.intensity = 0.6f;
            params.glow.blend_mode = 0; // Screen
            params.halation.intensity = 0.3f;
            params.grain.shadows_amount = 0.15f;
            params.grain.mids_amount = 0.15f;
            params.grain.highlights_amount = 0.15f;
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
            params.grain.shadows_amount = 0.35f;
            params.grain.mids_amount = 0.25f;
            params.grain.highlights_amount = 0.15f;
            params.grain.size = 1.5f;
            params.chromatic_aberration.amount = 0.2f;
            params.chromatic_aberration.blurriness = 2.0f;
            params.chromatic_aberration.red_scale = 1.1f;
            params.chromatic_aberration.green_scale = 1.0f;
            params.chromatic_aberration.blue_scale = 0.9f;
            return params;
        }

        inline EffectParameters DreamyDiffusion() {
            EffectParameters params;
            params.bloom.amount = 0.6f;
            params.bloom.radius = 60.0f;
            params.glow.threshold = 0.6f;
            params.glow.radius_x = 80.0f;
            params.glow.radius_y = 80.0f;
            params.glow.diffusion_radius = 80.0f;
            params.glow.intensity = 0.8f;
            params.glow.desaturation = 0.2f;
            params.grain.shadows_amount = 0.1f;
            params.grain.mids_amount = 0.15f;
            params.grain.highlights_amount = 0.05f;
            return params;
        }

        inline EffectParameters SubtleGrain() {
            EffectParameters params;
            params.grain.shadows_amount = 0.25f;
            params.grain.mids_amount = 0.15f;
            params.grain.highlights_amount = 0.1f;
            params.grain.size = 0.8f;
            params.grain.roughness = 0.3f;
            return params;
        }
    }

} // namespace CinematicFX

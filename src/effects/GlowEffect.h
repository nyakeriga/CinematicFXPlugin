/*******************************************************************************
 * CinematicFX - Glow Effect Implementation
 * 
 * Selective highlight diffusion (Pro-Mist style)
 ******************************************************************************/

#pragma once

#include "EffectBase.h"

namespace CinematicFX {

    /**
     * @brief Glow effect (highlight diffusion)
     * 
     * Algorithm:
     * 1. Extract highlights above threshold
     * 2. Apply Gaussian blur
     * 3. Additive blend with controlled intensity
     */
    class GlowEffect : public EffectBase {
    public:
        GlowEffect(const GlowParameters& params);

        const char* GetName() const override { return "Glow"; }
        bool IsActive() const override { return params_.intensity > 0.0f; }

        void Render(
            IGPUBackend* backend,
            GPUTexture input,
            GPUTexture output
        ) override;

        void SetParameters(const GlowParameters& params) { params_ = params; }

    private:
        GlowParameters params_;
    };

} // namespace CinematicFX

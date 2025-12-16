/*******************************************************************************
 * CinematicFX - Bloom Effect Implementation
 * 
 * Atmospheric bloom with shadow/midtone lift
 ******************************************************************************/

#pragma once

#include "EffectBase.h"

namespace CinematicFX {

    /**
     * @brief Bloom effect (atmospheric diffusion)
     * 
     * Algorithm:
     * 1. Extract luminance
     * 2. Apply shadow/midtone boost curve
     * 3. Separable Gaussian blur
     * 4. Additive blend with tint
     */
    class BloomEffect : public EffectBase {
    public:
        BloomEffect(const BloomParameters& params);

        const char* GetName() const override { return "Bloom"; }
        bool IsActive() const override { return params_.amount > 0.0f; }

        void Render(
            IGPUBackend* backend,
            GPUTexture input,
            GPUTexture output
        ) override;

        void SetParameters(const BloomParameters& params) { params_ = params; }

    private:
        BloomParameters params_;
    };

} // namespace CinematicFX

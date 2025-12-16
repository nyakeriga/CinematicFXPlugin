/*******************************************************************************
 * CinematicFX - Effect Base Class
 * 
 * Abstract base for all effect implementations
 ******************************************************************************/

#pragma once

#include "EffectParameters.h"
#include "GPUInterface.h"

namespace CinematicFX {

    /**
     * @brief Abstract base class for all effects
     */
    class EffectBase {
    public:
        virtual ~EffectBase() = default;

        /**
         * @brief Get effect name
         * @return Human-readable effect name
         */
        virtual const char* GetName() const = 0;

        /**
         * @brief Check if effect is active (has non-zero intensity)
         * @return true if effect should be applied
         */
        virtual bool IsActive() const = 0;

        /**
         * @brief Render this effect
         * @param backend GPU backend to use
         * @param input Input texture
         * @param output Output texture
         */
        virtual void Render(
            IGPUBackend* backend,
            GPUTexture input,
            GPUTexture output
        ) = 0;

    protected:
        // Helper: Blend two textures (additive blend)
        static void AdditiveBlend(
            IGPUBackend* backend,
            GPUTexture base,
            GPUTexture overlay,
            GPUTexture output,
            float strength
        );
    };

} // namespace CinematicFX

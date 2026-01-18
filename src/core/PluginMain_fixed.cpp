/*******************************************************************************
 * CinematicFX - Plugin Main Entry Point
 * 
 * Adobe After Effects / Premiere Pro SDK integration
 *******************************************************************************/
#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"

#include "../../include/CinematicFX.h"
#include "../../include/EffectParameters.h"
#include "../../include/GPUInterface.h"
#include "RenderPipeline.h"
#include "../utils/Logger.h"
#include "../utils/ColorConversion.h"
#include <memory>

#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <algorithm>

// Parameter IDs
enum {
    CINEMATICFX_INPUT = 0,
    
    // Master controls
    CINEMATICFX_OUTPUT_ENABLED,
    
    // Glow parameters
    CINEMATICFX_GLOW_ENABLED,
    CINEMATICFX_GLOW_THRESHOLD,
    CINEMATICFX_GLOW_INTENSITY,
    CINEMATICFX_GLOW_RADIUS_X,
    CINEMATICFX_GLOW_RADIUS_Y,
    CINEMATICFX_GLOW_DESATURATION,
    CINEMATICFX_GLOW_BLEND_MODE,
    CINEMATICFX_GLOW_TINT,

    // Halation parameters
    CINEMATICFX_HALATION_ENABLED,
    CINEMATICFX_HALATION_INTENSITY,
    CINEMATICFX_HALATION_RADIUS,
    CINEMATICFX_HALATION_HUE,
    CINEMATICFX_HALATION_SATURATION,
    CINEMATICFX_HALATION_THRESHOLD,

    // Grain parameters
    CINEMATICFX_GRAIN_ENABLED,
    CINEMATICFX_GRAIN_SHADOWS,
    CINEMATICFX_GRAIN_MIDS,
    CINEMATICFX_GRAIN_HIGHLIGHTS,
    CINEMATICFX_GRAIN_SIZE,
    CINEMATICFX_GRAIN_SOFTNESS,
    CINEMATICFX_GRAIN_SATURATION,

    // Chromatic Aberration parameters
    CINEMATICFX_CHROMA_ENABLED,
    CINEMATICFX_CHROMA_AMOUNT,
    CINEMATICFX_CHROMA_RED_SCALE,
    CINEMATICFX_CHROMA_GREEN_SCALE,
    CINEMATICFX_CHROMA_BLUE_SCALE,
    CINEMATICFX_CHROMA_BLURRINESS,
    CINEMATICFX_CHROMA_ANGLE,
    
    CINEMATICFX_NUM_PARAMS
};

// Global plugin instance data
struct GlobalData {
    std::unique_ptr<CinematicFX::RenderPipeline> render_pipeline;
    std::unique_ptr<CinematicFX::GPUContext> gpu_context;
    bool initialized = false;
};

static GlobalData g_global_data;

/*******************************************************************************
 * Parameter Extraction Helper
 *******************************************************************************/
static CinematicFX::EffectParameters ExtractParameters(PF_ParamDef** params) {
    CinematicFX::EffectParameters p;

    // Master enable
    p.output_enabled = params[CINEMATICFX_OUTPUT_ENABLED] ? params[CINEMATICFX_OUTPUT_ENABLED]->u.bd.value : true;

    // Glow enable and parameters (merged bloom+glow)
    bool glow_enabled = params[CINEMATICFX_GLOW_ENABLED] ? params[CINEMATICFX_GLOW_ENABLED]->u.bd.value : true;
    p.glow.intensity = params[CINEMATICFX_GLOW_INTENSITY] ? params[CINEMATICFX_GLOW_INTENSITY]->u.fs_d.value / 100.0f : 0.0f;
    p.glow.threshold = params[CINEMATICFX_GLOW_THRESHOLD] ? params[CINEMATICFX_GLOW_THRESHOLD]->u.fs_d.value / 100.0f : 0.0f;
    p.glow.diffusion_radius = params[CINEMATICFX_GLOW_RADIUS_X] ? params[CINEMATICFX_GLOW_RADIUS_X]->u.fs_d.value : 0.0f;
    p.glow.desaturation = params[CINEMATICFX_GLOW_DESATURATION] ? params[CINEMATICFX_GLOW_DESATURATION]->u.fs_d.value / 100.0f : 0.0f;
    p.glow.blend_mode = params[CINEMATICFX_GLOW_BLEND_MODE] ? params[CINEMATICFX_GLOW_BLEND_MODE]->u.pd.value : 0;
    PF_Pixel* tint_pixel = params[CINEMATICFX_GLOW_TINT] ? &params[CINEMATICFX_GLOW_TINT]->u.cd.value : nullptr;
    if (tint_pixel) {
        p.glow.tint_r = static_cast<float>(tint_pixel->red) / 255.0f;
        p.glow.tint_g = static_cast<float>(tint_pixel->green) / 255.0f;
        p.glow.tint_b = static_cast<float>(tint_pixel->blue) / 255.0f;
    } else {
        p.glow.tint_r = 1.0f;
        p.glow.tint_g = 1.0f;
        p.glow.tint_b = 1.0f;
    }
    if (!glow_enabled) p.glow.intensity = 0.0f;

    // Halation parameters
    p.halation.enabled = params[CINEMATICFX_HALATION_ENABLED] ? params[CINEMATICFX_HALATION_ENABLED]->u.bd.value : true;
    p.halation.intensity = params[CINEMATICFX_HALATION_INTENSITY] ? params[CINEMATICFX_HALATION_INTENSITY]->u.fs_d.value / 100.0f : 0.0f;
    p.halation.spread = params[CINEMATICFX_HALATION_RADIUS] ? params[CINEMATICFX_HALATION_RADIUS]->u.fs_d.value : 0.0f;
    p.halation.hue = params[CINEMATICFX_HALATION_HUE] ? params[CINEMATICFX_HALATION_HUE]->u.fs_d.value : 0.0f;
    p.halation.saturation = params[CINEMATICFX_HALATION_SATURATION] ? params[CINEMATICFX_HALATION_SATURATION]->u.fs_d.value / 100.0f : 1.0f;
    p.halation.threshold = params[CINEMATICFX_HALATION_THRESHOLD] ? params[CINEMATICFX_HALATION_THRESHOLD]->u.fs_d.value / 100.0f : 0.5f;

    // Grain parameters
    p.grain.enabled = params[CINEMATICFX_GRAIN_ENABLED] ? params[CINEMATICFX_GRAIN_ENABLED]->u.bd.value : true;
    p.grain.shadows_amount = params[CINEMATICFX_GRAIN_SHADOWS] ? params[CINEMATICFX_GRAIN_SHADOWS]->u.fs_d.value / 100.0f : 0.0f;
    p.grain.mids_amount = params[CINEMATICFX_GRAIN_MIDS] ? params[CINEMATICFX_GRAIN_MIDS]->u.fs_d.value / 100.0f : 0.0f;
    p.grain.highlights_amount = params[CINEMATICFX_GRAIN_HIGHLIGHTS] ? params[CINEMATICFX_GRAIN_HIGHLIGHTS]->u.fs_d.value / 100.0f : 0.0f;
    p.grain.size = params[CINEMATICFX_GRAIN_SIZE] ? params[CINEMATICFX_GRAIN_SIZE]->u.fs_d.value : 1.0f;
    p.grain.roughness = params[CINEMATICFX_GRAIN_SOFTNESS] ? params[CINEMATICFX_GRAIN_SOFTNESS]->u.fs_d.value / 100.0f : 0.0f;
    p.grain.saturation = params[CINEMATICFX_GRAIN_SATURATION] ? params[CINEMATICFX_GRAIN_SATURATION]->u.fs_d.value / 100.0f : 1.0f;
    p.grain.amount = (p.grain.shadows_amount + p.grain.mids_amount + p.grain.highlights_amount) / 3.0f;

    // Chromatic Aberration parameters
    p.chromatic_aberration.enabled = params[CINEMATICFX_CHROMA_ENABLED] ? params[CINEMATICFX_CHROMA_ENABLED]->u.bd.value : true;
    p.chromatic_aberration.amount = params[CINEMATICFX_CHROMA_AMOUNT] ? params[CINEMATICFX_CHROMA_AMOUNT]->u.fs_d.value / 100.0f : 0.0f;
    p.chromatic_aberration.red_scale = params[CINEMATICFX_CHROMA_RED_SCALE] ? params[CINEMATICFX_CHROMA_RED_SCALE]->u.fs_d.value : 1.0f;
    p.chromatic_aberration.green_scale = params[CINEMATICFX_CHROMA_GREEN_SCALE] ? params[CINEMATICFX_CHROMA_GREEN_SCALE]->u.fs_d.value : 1.0f;
    p.chromatic_aberration.blue_scale = params[CINEMATICFX_CHROMA_BLUE_SCALE] ? params[CINEMATICFX_CHROMA_BLUE_SCALE]->u.fs_d.value : 1.0f;
    p.chromatic_aberration.blurriness = params[CINEMATICFX_CHROMA_BLURRINESS] ? params[CINEMATICFX_CHROMA_BLURRINESS]->u.fs_d.value : 0.0f;
    p.chromatic_aberration.angle = params[CINEMATICFX_CHROMA_ANGLE] ? params[CINEMATICFX_CHROMA_ANGLE]->u.ad.value : 0.0f;

    // Validate and clamp to safe ranges to avoid NaNs and overflows
    p.ValidateAll();
    return p;
}

/*******************************************************************************
 * Global Setup - Initialize plugin
 *******************************************************************************/
static PF_Err GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    // Set plugin info
    out_data->my_version = PF_VERSION(
        CINEMATICFX_VERSION_MAJOR,
        CINEMATICFX_VERSION_MINOR,
        0,
        0,
        0
    );

    out_data->name[0] = '\0';
    strncpy(out_data->name, "CinematicFX", sizeof(out_data->name) - 1);

    // Only advertise features that are actually implemented (SAFE BASELINE)
    out_data->out_flags =
        PF_OutFlag_PIX_INDEPENDENT;

    out_data->out_flags2 = 0;

    // Initialize GPU context with automatic backend selection
    g_global_data.gpu_context = CinematicFX::GPUContext::Create();
    g_global_data.initialized = true;

    return PF_Err_NONE;
}

/*******************************************************************************
 * Global Setdown - Cleanup plugin
 *******************************************************************************/
static PF_Err GlobalSetdown(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    if (g_global_data.initialized) {
        g_global_data.render_pipeline.reset();
        g_global_data.gpu_context.reset();
        CinematicFX::Logger::Shutdown();
        g_global_data.initialized = false;
    }
    
    return PF_Err_NONE;
}

/*******************************************************************************
 * Params Setup - Define all parameters
 *******************************************************************************/
static PF_Err ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    PF_Err err = PF_Err_NONE;
    
    // ENHANCED: Critical NULL checks with comprehensive validation
    if (!in_data || !out_data) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    
    PF_ParamDef def;
    
    // Master Output Enable
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Output", "", FALSE, 0, CINEMATICFX_OUTPUT_ENABLED);

    // Bloom group removed (merged into Glow)

    // --- GLOW PARAMETERS ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Glow", "", FALSE, 0, CINEMATICFX_GLOW_ENABLED);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Glow Threshold", 0, 100, 0, 100, 70, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_THRESHOLD);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Glow Intensity", 0, 200, 0, 200, 100, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_INTENSITY);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Glow Radius X", 1, 100, 1, 100, 20, PF_Precision_TENTHS, 0, 0, CINEMATICFX_GLOW_RADIUS_X);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Glow Radius Y", 1, 100, 1, 100, 20, PF_Precision_TENTHS, 0, 0, CINEMATICFX_GLOW_RADIUS_Y);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Glow Desaturation", 0, 100, 0, 100, 30, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_DESATURATION);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Glow Blend Mode", 3, 1, "Screen|Add|Normal", CINEMATICFX_GLOW_BLEND_MODE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Glow Tint", 255, 255, 255, CINEMATICFX_GLOW_TINT);
    
    // --- HALATION PARAMETERS ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Halation", "", TRUE, 0, CINEMATICFX_HALATION_ENABLED);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Halation Intensity", 0, 100, 0, 100, 60, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_INTENSITY);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Halation Radius", 1, 50, 1, 50, 15, PF_Precision_TENTHS, 0, 0, CINEMATICFX_HALATION_RADIUS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Halation Hue", 0, 360, 0, 360, 0, PF_Precision_TENTHS, 0, 0, CINEMATICFX_HALATION_HUE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Halation Saturation", 0, 200, 0, 200, 100, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_SATURATION);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Halation Threshold", 0, 100, 0, 100, 30, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_THRESHOLD);
    
    // --- GRAIN PARAMETERS ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Grain", "", TRUE, 0, CINEMATICFX_GRAIN_ENABLED);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Shadows Grain", 0, 100, 0, 100, 20, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SHADOWS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Midtones Grain", 0, 100, 0, 100, 35, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_MIDS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Highlights Grain", 0, 100, 0, 100, 15, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_HIGHLIGHTS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Grain Size", 0.5, 5.0, 0.5, 5.0, 1.0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SIZE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Grain Softness", 0, 100, 0, 100, 50, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SOFTNESS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Grain Saturation", 0, 200, 0, 200, 100, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SATURATION);
    
    // --- CHROMATIC ABERRATION PARAMETERS ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Chromatic Aberration", "", TRUE, 0, CINEMATICFX_CHROMA_ENABLED);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Chromatic Aberration Amount", 0, 100, 0, 100, 10, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_CHROMA_AMOUNT);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Red Channel Scale", 0.5, 2.0, 0.5, 2.0, 1.0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_CHROMA_RED_SCALE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Green Channel Scale", 0.5, 2.0, 0.5, 2.0, 1.0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_CHROMA_GREEN_SCALE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blue Channel Scale", 0.5, 2.0, 0.5, 2.0, 1.0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_CHROMA_BLUE_SCALE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blurriness", 0, 10, 0, 10, 0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_CHROMA_BLURRINESS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Angle", 0, CINEMATICFX_CHROMA_ANGLE);
    
    out_data->num_params = CINEMATICFX_NUM_PARAMS;
    
    return PF_Err_NONE;
}

/*******************************************************************************
 * Render - Main rendering function
 *******************************************************************************/
static PF_Err Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    if (!in_data || !out_data || !params || !output) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    PF_LayerDef* input = &params[CINEMATICFX_INPUT]->u.ld;
    if (!input->data || !output->data) {
        return PF_Err_NONE;
    }

    int width = std::min(input->width, output->width);
    int height = std::min(input->height, output->height);

    // Extract effect parameters
    CinematicFX::EffectParameters effect_params = ExtractParameters(params);

    // Fast path: direct copy without processing if no effects active
    if (!effect_params.output_enabled || !effect_params.HasActiveEffects()) {
        int width = std::min(input->width, output->width);
        int height = std::min(input->height, output->height);
        size_t copy_bytes = std::min(static_cast<size_t>(input->rowbytes) * height,
                                   static_cast<size_t>(output->rowbytes) * height);
        memcpy(output->data, input->data, copy_bytes);
        return PF_Err_NONE;
    }

    // Convert Adobe PF_Pixel8 → Engine Float Buffer
    auto input_float = CinematicFX::ColorConversion::AdobeToEngine(
        (CinematicFX::PF_Pixel8*)input->data,
        width,
        height,
        input->rowbytes
    );

    // Create FrameBuffer for RenderPipeline
    CinematicFX::FrameBuffer input_frame;
    input_frame.data = input_float.get();
    input_frame.width = width;
    input_frame.height = height;
    input_frame.stride = width * 4; // RGBA float
    input_frame.owns_data = false; // Managed by unique_ptr

    // Allocate output FrameBuffer
    auto output_float = CinematicFX::ColorConversion::CreateEngineBuffer(width, height);
    CinematicFX::FrameBuffer output_frame;
    output_frame.data = output_float.get();
    output_frame.width = width;
    output_frame.height = height;
    output_frame.stride = width * 4;
    output_frame.owns_data = false;

    // Initialize RenderPipeline if needed
    if (!g_global_data.render_pipeline && g_global_data.gpu_context) {
        g_global_data.render_pipeline = std::make_unique<CinematicFX::RenderPipeline>(g_global_data.gpu_context.get());
    }

    // Process through RenderPipeline (Float Engine)
    uint32_t frame_number = 0; // TODO: Get actual frame number from AE
    bool success = false;

    if (g_global_data.render_pipeline) {
        success = g_global_data.render_pipeline->RenderFrame(
            input_frame,
            output_frame,
            effect_params,
            frame_number
        );
    }

    if (!success) {
        // Fallback: just copy input to output
        memcpy(output->data, input->data,
               std::min(input->rowbytes * height, output->rowbytes * height));
        return PF_Err_NONE;
    }

    // Convert Engine Float Buffer → Adobe PF_Pixel8
    CinematicFX::ColorConversion::EngineToAdobe(
        output_float.get(),
        (CinematicFX::PF_Pixel8*)output->data,
        width,
        height,
        output->rowbytes
    );

    return PF_Err_NONE;
}

/*******************************************************************************
 * About - Display plugin information
 *******************************************************************************/
static PF_Err About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    // Simple About message
    sprintf_s(
        out_data->return_msg,
        sizeof(out_data->return_msg),
        "%s v%d.%d\\r\\rCinematic Film Effects\\r"
        "Professional Bloom, Glow, Halation, Grain & Chromatic Aberration\\r\\r"
        "(c) 2025 Pol Casals\\r"
        "GPU-Accelerated for Maximum Performance",
        "CinematicFX",
        CINEMATICFX_VERSION_MAJOR,
        CINEMATICFX_VERSION_MINOR
    );
    
    return PF_Err_NONE;
}

/*******************************************************************************
 * Entry Point Function
 *******************************************************************************/
DllExport PF_Err PluginDataEntryFunction(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion
) {
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT(
        inPtr,
        inPluginDataCallBackPtr,
        "CinematicFX",              // Name
        "com.cinebloom.cinematicfx",// Match Name (unique, non-Adobe)
        "Channel",                  // Category (standard Premiere category)
        AE_RESERVED_INFO            // Reserved
    );
    // Explicit host technology (PremierePro)
    // This may require a metadata file or registration macro depending on SDK version
    // If supported, set pluginHostTechnology: "PremierePro" in descriptor
    return result;
}

PF_Err EffectMain(
    PF_Cmd cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output,
    void* extra
) {
    PF_Err err = PF_Err_NONE;
    
    // CRITICAL: Validate input pointers before ANY processing
    if (!in_data || !out_data) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data, params, output);
                break;
                
            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data, params, output);
                break;
                
            case PF_Cmd_GLOBAL_SETDOWN:
                err = GlobalSetdown(in_data, out_data, params, output);
                break;
                
            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data, params, output);
                break;
                
            case PF_Cmd_SEQUENCE_SETUP:
                // Premiere-specific: setup for sequence
                out_data->sequence_data = nullptr;
                err = PF_Err_NONE;
                break;
                
            case PF_Cmd_SEQUENCE_SETDOWN:
                // Premiere-specific: cleanup for sequence
                if (out_data && out_data->sequence_data) {
                    out_data->sequence_data = nullptr;
                }
                err = PF_Err_NONE;
                break;
                
            case PF_Cmd_RENDER:
                if (params && output) {
                    err = Render(in_data, out_data, params, output);
                } else {
                    err = PF_Err_BAD_CALLBACK_PARAM;
                }
                break;
                
            default:
                // Unknown command - safe to ignore
                err = PF_Err_NONE;
                break;
        }
    }
    catch (const PF_Err& thrown_err) {
        // Handle PF_Err exceptions
        err = thrown_err;
    }
    catch (const std::bad_alloc&) {
        // Memory allocation failure
        err = PF_Err_OUT_OF_MEMORY;
    }
    catch (const std::exception& e) {
        // Standard C++ exceptions
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    catch (...) {
        // Catch ALL other exceptions to prevent crashes
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    
    return err;
}
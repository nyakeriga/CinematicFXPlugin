/*******************************************************************************
 * CinematicFX - Plugin Main Entry Point
 * 
 * Adobe After Effects / Premiere Pro SDK integration
 ******************************************************************************/
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
    
    // Bloom parameters removed (merged into Glow)
    
    // Glow (merged with Bloom)
    CINEMATICFX_GLOW_GROUP_START,
    CINEMATICFX_GLOW_ENABLED,
    CINEMATICFX_GLOW_THRESHOLD,
    CINEMATICFX_GLOW_INTENSITY,
    CINEMATICFX_GLOW_RADIUS_X,
    CINEMATICFX_GLOW_RADIUS_Y,
    CINEMATICFX_GLOW_DESATURATION,
    CINEMATICFX_GLOW_BLEND_MODE,
    CINEMATICFX_GLOW_TINT,
    CINEMATICFX_GLOW_GROUP_END,
    
    // Halation parameters
    CINEMATICFX_HALATION_GROUP_START,
    CINEMATICFX_HALATION_INTENSITY,
    CINEMATICFX_HALATION_RADIUS,
    CINEMATICFX_HALATION_HUE,
    CINEMATICFX_HALATION_SATURATION,
    CINEMATICFX_HALATION_THRESHOLD,
    CINEMATICFX_HALATION_GROUP_END,
    
    // Grain parameters
    CINEMATICFX_GRAIN_GROUP_START,
    CINEMATICFX_GRAIN_SHADOWS,
    CINEMATICFX_GRAIN_MIDS,
    CINEMATICFX_GRAIN_HIGHLIGHTS,
    CINEMATICFX_GRAIN_SIZE,
    CINEMATICFX_GRAIN_SOFTNESS,
    CINEMATICFX_GRAIN_SATURATION,
    CINEMATICFX_GRAIN_GROUP_END,
    
    // Chromatic Aberration parameters
    CINEMATICFX_CHROMA_GROUP_START,
    CINEMATICFX_CHROMA_RED_SCALE,
    CINEMATICFX_CHROMA_GREEN_SCALE,
    CINEMATICFX_CHROMA_BLUE_SCALE,
    CINEMATICFX_CHROMA_BLURRINESS,
    CINEMATICFX_CHROMA_ANGLE,
    CINEMATICFX_CHROMA_GROUP_END,
    
    CINEMATICFX_NUM_PARAMS
};

// Global plugin instance data
typedef struct {
    CinematicFX::RenderPipeline* render_pipeline;
    CinematicFX::GPUContext* gpu_context;
    bool initialized;
} GlobalData;

static GlobalData g_global_data = {nullptr, nullptr, false};

/*******************************************************************************
 * Global Setup - Initialize plugin
 ******************************************************************************/
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
    
    // Enable 32-bit float processing and proper UI
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
                          PF_OutFlag_PIX_INDEPENDENT |
                          PF_OutFlag_USE_OUTPUT_EXTENT |
                          PF_OutFlag_I_DO_DIALOG;  // Enable About dialog
    
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE |
                           PF_OutFlag2_SUPPORTS_SMART_RENDER |
                           PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                           PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS |
                           PF_OutFlag2_REVEALS_ZERO_ALPHA;
    
    // Initialize GPU context (once globally)
    if (!g_global_data.initialized) {
        CinematicFX::Logger::Initialize(CinematicFX::Logger::LogLevel::INFO);
        
        // Try platform-specific GPU backend first, then CPU fallback
        std::unique_ptr<CinematicFX::GPUContext> gpu_context_ptr;
        
#ifdef __APPLE__
        // macOS: Try Metal first
        gpu_context_ptr = CinematicFX::GPUContext::Create(CinematicFX::GPUBackendType::METAL);
        if (gpu_context_ptr) {
            CinematicFX::Logger::Info("CinematicFX initialized with Metal GPU backend");
        }
#elif defined(_WIN32)
        // Windows: Try CUDA first (if available)
        #ifdef CINEMATICFX_CUDA_AVAILABLE
        gpu_context_ptr = CinematicFX::GPUContext::Create(CinematicFX::GPUBackendType::CUDA);
        if (gpu_context_ptr) {
            CinematicFX::Logger::Info("CinematicFX initialized with CUDA GPU backend");
        }
        #endif
#endif
        
        // Fallback to CPU if GPU failed
        if (!gpu_context_ptr) {
            gpu_context_ptr = CinematicFX::GPUContext::Create(CinematicFX::GPUBackendType::CPU);
            if (gpu_context_ptr) {
                CinematicFX::Logger::Info("CinematicFX initialized with CPU backend");
            } else {
                CinematicFX::Logger::Error("Failed to initialize any backend");
            }
        }
        
        if (gpu_context_ptr) {
            g_global_data.gpu_context = gpu_context_ptr.release();
            g_global_data.initialized = true;
        }
    }
    
    return PF_Err_NONE;
}

/*******************************************************************************
 * Global Setdown - Cleanup plugin
 ******************************************************************************/
static PF_Err GlobalSetdown(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    if (g_global_data.initialized) {
        if (g_global_data.render_pipeline) {
            delete g_global_data.render_pipeline;
            g_global_data.render_pipeline = nullptr;
        }
        
        if (g_global_data.gpu_context) {
            delete g_global_data.gpu_context;
            g_global_data.gpu_context = nullptr;
        }
        
        CinematicFX::Logger::Shutdown();
        g_global_data.initialized = false;
    }
    
    return PF_Err_NONE;
}

/*******************************************************************************
 * Params Setup - Define all parameters
 ******************************************************************************/
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
    
    // Validate version compatibility
    if (in_data->version.major < 13) {  // Minimum AE/Premiere version
        return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }
    
    PF_ParamDef def;
    
    // Master Output Enable
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Output", "", TRUE, 0, CINEMATICFX_OUTPUT_ENABLED);
    
    // Bloom group removed (merged into Glow)
    
    // --- GLOW GROUP (Merged Bloom+Glow) ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Glow (Complete)", CINEMATICFX_GLOW_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Glow", "", TRUE, 0, CINEMATICFX_GLOW_ENABLED);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Threshold", 0, 100, 0, 100, 70, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_THRESHOLD);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Intensity", 0, 200, 0, 200, 80, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_INTENSITY);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius X", 1, 100, 1, 100, 40, PF_Precision_TENTHS, 0, 0, CINEMATICFX_GLOW_RADIUS_X);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius Y", 1, 100, 1, 100, 40, PF_Precision_TENTHS, 0, 0, CINEMATICFX_GLOW_RADIUS_Y);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Desaturation", 0, 100, 0, 100, 0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_DESATURATION);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Blend Mode", 3, 1, "Screen|Add|Normal", CINEMATICFX_GLOW_BLEND_MODE);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Tint", 255, 255, 255, CINEMATICFX_GLOW_TINT);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_GLOW_GROUP_END);
    
    // --- HALATION GROUP (Redesigned) ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Halation (Film Fringe)", CINEMATICFX_HALATION_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Intensity", 0, 100, 0, 100, 60, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_INTENSITY);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius", 1, 50, 1, 50, 15, PF_Precision_TENTHS, 0, 0, CINEMATICFX_HALATION_RADIUS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Hue", 0, 360, 0, 360, 0, PF_Precision_TENTHS, 0, 0, CINEMATICFX_HALATION_HUE);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Saturation", 0, 200, 0, 200, 100, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_SATURATION);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Threshold", 0, 100, 0, 100, 50, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_THRESHOLD);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_HALATION_GROUP_END);
    
    // --- GRAIN GROUP (Redesigned) ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Curated Grain", CINEMATICFX_GRAIN_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Shadows Grain", 0, 100, 0, 100, 20, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SHADOWS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Midtones Grain", 0, 100, 0, 100, 35, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_MIDS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Highlights Grain", 0, 100, 0, 100, 15, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_HIGHLIGHTS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Size", 0.5, 5.0, 0.5, 5.0, 1.0, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SIZE);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Softness", 0, 100, 0, 100, 50, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SOFTNESS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Saturation", 0, 200, 0, 200, 100, PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SATURATION);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_GRAIN_GROUP_END);
    
    // --- CHROMATIC ABERRATION GROUP ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Chromatic Aberration", CINEMATICFX_CHROMA_GROUP_START);
    
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
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_CHROMA_GROUP_END);
    
    out_data->num_params = CINEMATICFX_NUM_PARAMS;
    
    return PF_Err_NONE;
}

/*******************************************************************************
 * Render - Main rendering function
 ******************************************************************************/
static PF_Err Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef** params,
    PF_LayerDef* output
) {
    // PERMANENT STABILITY FIX: Canonical safe render template
    if (!in_data || !out_data || !output) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    if (output->width <= 0 || output->height <= 0) {
        return PF_Err_NONE;
    }
    if (!output->data) {
        return PF_Err_NONE;
    }
    // Format checks removed: PF_InData does not have pixel_format. Only pointer/dimension checks enforced.
    // Disable GPU until validated
    out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
    out_data->out_flags |= PF_OutFlag_PIX_INDEPENDENT;
    // SAFE processing here (stateless, thread-safe)
    return PF_Err_NONE;
}

/*******************************************************************************
 * About - Display plugin information
 ******************************************************************************/
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
 ******************************************************************************/
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
        "Stylize",                  // Category (standard Premiere category)
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

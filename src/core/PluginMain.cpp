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
    
    // Bloom parameters
    CINEMATICFX_BLOOM_GROUP_START,
    CINEMATICFX_BLOOM_AMOUNT,
    CINEMATICFX_BLOOM_RADIUS,
    CINEMATICFX_BLOOM_TINT,
    CINEMATICFX_BLOOM_GROUP_END,
    
    // Glow parameters
    CINEMATICFX_GLOW_GROUP_START,
    CINEMATICFX_GLOW_THRESHOLD,
    CINEMATICFX_GLOW_RADIUS,
    CINEMATICFX_GLOW_INTENSITY,
    CINEMATICFX_GLOW_GROUP_END,
    
    // Halation parameters
    CINEMATICFX_HALATION_GROUP_START,
    CINEMATICFX_HALATION_INTENSITY,
    CINEMATICFX_HALATION_RADIUS,
    CINEMATICFX_HALATION_GROUP_END,
    
    // Grain parameters
    CINEMATICFX_GRAIN_GROUP_START,
    CINEMATICFX_GRAIN_AMOUNT,
    CINEMATICFX_GRAIN_SIZE,
    CINEMATICFX_GRAIN_LUMA_MAPPING,
    CINEMATICFX_GRAIN_GROUP_END,
    
    // Chromatic Aberration parameters
    CINEMATICFX_CHROMA_GROUP_START,
    CINEMATICFX_CHROMA_AMOUNT,
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
    
    // --- BLOOM GROUP ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Bloom", CINEMATICFX_BLOOM_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Amount", 0, 100, 0, 100, 50, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_BLOOM_AMOUNT);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius", 1, 100, 1, 100, 40, 
                         PF_Precision_TENTHS, 0, 0, CINEMATICFX_BLOOM_RADIUS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Tint", 255, 255, 255, CINEMATICFX_BLOOM_TINT);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_BLOOM_GROUP_END);
    
    // --- GLOW GROUP ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Glow (Pro-Mist)", CINEMATICFX_GLOW_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Threshold", 0, 100, 0, 100, 70, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_THRESHOLD);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius", 1, 100, 1, 100, 40, 
                         PF_Precision_TENTHS, 0, 0, CINEMATICFX_GLOW_RADIUS);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Intensity", 0, 200, 0, 200, 80, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GLOW_INTENSITY);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_GLOW_GROUP_END);
    
    // --- HALATION GROUP ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Halation (Film Fringe)", CINEMATICFX_HALATION_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Intensity", 0, 100, 0, 100, 60, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_HALATION_INTENSITY);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius", 1, 50, 1, 50, 15, 
                         PF_Precision_TENTHS, 0, 0, CINEMATICFX_HALATION_RADIUS);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_HALATION_GROUP_END);
    
    // --- GRAIN GROUP ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Curated Grain", CINEMATICFX_GRAIN_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Amount", 0, 100, 0, 100, 35, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_AMOUNT);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Size", 0.5, 5.0, 0.5, 5.0, 1.0, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_SIZE);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Luma Mapping", 0, 100, 0, 100, 50, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_GRAIN_LUMA_MAPPING);
    
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CINEMATICFX_GRAIN_GROUP_END);
    
    // --- CHROMATIC ABERRATION GROUP ---
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Chromatic Aberration", CINEMATICFX_CHROMA_GROUP_START);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Amount", 0, 10, 0, 10, 0, 
                         PF_Precision_HUNDREDTHS, 0, 0, CINEMATICFX_CHROMA_AMOUNT);
    
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
    PF_Err err = PF_Err_NONE;
    
    // ENHANCED: Critical NULL and validity checks with detailed validation
    if (!in_data || !out_data || !params || !output) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    
    // Validate input layer
    if (!params[CINEMATICFX_INPUT]) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    
    // Validate layer data structure
    PF_LayerDef* input_layer = &params[CINEMATICFX_INPUT]->u.ld;
    if (!input_layer || !input_layer->data || input_layer->width <= 0 || input_layer->height <= 0) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    
    // Validate output layer
    if (!output->data || output->width <= 0 || output->height <= 0) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    
    // Check if output is enabled
    PF_ParamDef output_enabled_param;
    AEFX_CLR_STRUCT(output_enabled_param);
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_OUTPUT_ENABLED, 
                          in_data->current_time, 
                          in_data->time_step, 
                          in_data->time_scale, 
                          &output_enabled_param));
    
    if (!output_enabled_param.u.bd.value) {
        // Output disabled - just copy input to output
        ERR(PF_COPY(&params[CINEMATICFX_INPUT]->u.ld, output, NULL, NULL));
        ERR(PF_CHECKIN_PARAM(in_data, &output_enabled_param));
        return err;
    }
    ERR(PF_CHECKIN_PARAM(in_data, &output_enabled_param));
    
    // Get all parameters
    CinematicFX::EffectParameters effect_params;
    
    // Bloom
    PF_ParamDef bloom_amount, bloom_radius, bloom_tint;
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_BLOOM_AMOUNT, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &bloom_amount));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_BLOOM_RADIUS, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &bloom_radius));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_BLOOM_TINT, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &bloom_tint));
    
    // Safe conversion with bounds checking
    effect_params.bloom.amount = std::max(0.0f, std::min(1.0f, static_cast<float>(bloom_amount.u.fs_d.value) / 100.0f));
    effect_params.bloom.radius = std::max(1.0f, std::min(100.0f, static_cast<float>(bloom_radius.u.fs_d.value)));
    effect_params.bloom.tint_r = std::max(0.0f, std::min(1.0f, static_cast<float>(bloom_tint.u.cd.value.red) / 255.0f));
    effect_params.bloom.tint_g = std::max(0.0f, std::min(1.0f, static_cast<float>(bloom_tint.u.cd.value.green) / 255.0f));
    effect_params.bloom.tint_b = std::max(0.0f, std::min(1.0f, static_cast<float>(bloom_tint.u.cd.value.blue) / 255.0f));
    
    ERR(PF_CHECKIN_PARAM(in_data, &bloom_amount));
    ERR(PF_CHECKIN_PARAM(in_data, &bloom_radius));
    ERR(PF_CHECKIN_PARAM(in_data, &bloom_tint));
    
    // Glow
    PF_ParamDef glow_threshold, glow_radius, glow_intensity;
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_GLOW_THRESHOLD, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &glow_threshold));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_GLOW_RADIUS, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &glow_radius));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_GLOW_INTENSITY, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &glow_intensity));
    
    effect_params.glow.threshold = glow_threshold.u.fs_d.value / 100.0f;
    effect_params.glow.diffusion_radius = glow_radius.u.fs_d.value;
    effect_params.glow.intensity = glow_intensity.u.fs_d.value / 100.0f;
    
    ERR(PF_CHECKIN_PARAM(in_data, &glow_threshold));
    ERR(PF_CHECKIN_PARAM(in_data, &glow_radius));
    ERR(PF_CHECKIN_PARAM(in_data, &glow_intensity));
    
    // Halation
    PF_ParamDef halation_intensity, halation_radius;
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_HALATION_INTENSITY, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &halation_intensity));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_HALATION_RADIUS, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &halation_radius));
    
    effect_params.halation.intensity = halation_intensity.u.fs_d.value / 100.0f;
    effect_params.halation.spread = halation_radius.u.fs_d.value;
    
    ERR(PF_CHECKIN_PARAM(in_data, &halation_intensity));
    ERR(PF_CHECKIN_PARAM(in_data, &halation_radius));
    
    // Grain
    PF_ParamDef grain_amount, grain_size, grain_luma;
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_GRAIN_AMOUNT, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &grain_amount));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_GRAIN_SIZE, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &grain_size));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_GRAIN_LUMA_MAPPING, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &grain_luma));
    
    effect_params.grain.amount = grain_amount.u.fs_d.value / 100.0f;
    effect_params.grain.size = grain_size.u.fs_d.value;
    effect_params.grain.roughness = grain_luma.u.fs_d.value / 100.0f;
    
    ERR(PF_CHECKIN_PARAM(in_data, &grain_amount));
    ERR(PF_CHECKIN_PARAM(in_data, &grain_size));
    ERR(PF_CHECKIN_PARAM(in_data, &grain_luma));
    
    // Chromatic Aberration
    PF_ParamDef chroma_amount, chroma_angle;
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_CHROMA_AMOUNT, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &chroma_amount));
    ERR(PF_CHECKOUT_PARAM(in_data, CINEMATICFX_CHROMA_ANGLE, in_data->current_time, 
                          in_data->time_step, in_data->time_scale, &chroma_angle));
    
    effect_params.chromatic_aberration.amount = chroma_amount.u.fs_d.value;
    effect_params.chromatic_aberration.angle = chroma_angle.u.ad.value;
    
    ERR(PF_CHECKIN_PARAM(in_data, &chroma_amount));
    ERR(PF_CHECKIN_PARAM(in_data, &chroma_angle));
    
    // Validate all parameters
    effect_params.ValidateAll();
    
    // Create render pipeline if needed
    if (!g_global_data.render_pipeline && g_global_data.gpu_context) {
        try {
            g_global_data.render_pipeline = new CinematicFX::RenderPipeline(g_global_data.gpu_context);
            CinematicFX::Logger::Info("Render pipeline created successfully");
        } catch (const std::exception& e) {
            CinematicFX::Logger::Error("Failed to create render pipeline: %s", e.what());
            ERR(PF_COPY(&params[CINEMATICFX_INPUT]->u.ld, output, NULL, NULL));
            return err;
        }
    }
    
    if (g_global_data.render_pipeline && g_global_data.gpu_context) {
        // Convert AE buffers to our format
        PF_LayerDef* input_layer = &params[CINEMATICFX_INPUT]->u.ld;
        
        CinematicFX::FrameBuffer input_buffer;
        input_buffer.width = input_layer->width;
        input_buffer.height = input_layer->height;
        // Stride is in pixels (RGBA), rowbytes is total bytes per row
        input_buffer.stride = input_layer->width * 4; // RGBA
        input_buffer.data = reinterpret_cast<float*>(input_layer->data);
        input_buffer.owns_data = false;
        
        CinematicFX::FrameBuffer output_buffer;
        output_buffer.width = output->width;
        output_buffer.height = output->height;
        output_buffer.stride = output->width * 4; // RGBA
        output_buffer.data = reinterpret_cast<float*>(output->data);
        output_buffer.owns_data = false;
        
        // Render
        uint32_t frame_number = static_cast<uint32_t>(in_data->current_time);
        bool render_success = false;
        
        try {
            render_success = g_global_data.render_pipeline->RenderFrame(
                input_buffer, output_buffer, effect_params, frame_number);
        } catch (const std::exception& e) {
            CinematicFX::Logger::Error("Render exception: %s", e.what());
            render_success = false;
        }
        
        if (!render_success) {
            // On error, copy input to output
            CinematicFX::Logger::Warning("Rendering failed, copying input to output");
            ERR(PF_COPY(&params[CINEMATICFX_INPUT]->u.ld, output, NULL, NULL));
        }
    } else {
        // No pipeline - just copy input to output
        CinematicFX::Logger::Warning("No render pipeline available, copying input");
        ERR(PF_COPY(&params[CINEMATICFX_INPUT]->u.ld, output, NULL, NULL));
    }
    
    return err;
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
        "ADBE CinematicFX",         // Match Name (ADBE prefix for Adobe)
        "Stylize",                  // Category (standard Premiere category)
        AE_RESERVED_INFO            // Reserved
    );
    
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

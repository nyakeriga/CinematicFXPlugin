/*******************************************************************************
 * CinematicFX - GPU Context Implementation
 * 
 * Automatic backend selection and management
 ******************************************************************************/

#include "GPUInterface.h"
#include "CUDABackend.h"
#include "MetalBackend.h"
#include "CPUFallback.h"
#include "../utils/Logger.h"
#include <memory>

namespace CinematicFX {

    std::unique_ptr<GPUContext> GPUContext::Create(GPUBackendType preferred_backend) {
        auto context = std::unique_ptr<GPUContext>(new GPUContext());
        
        // Try preferred backend first
        GPUBackendType backend_to_try = preferred_backend;
        
        // Fallback chain: CUDA/Metal → CPU
        while (true) {
            context->backend_ = CreateBackend(backend_to_try);
            
            if (context->backend_ && context->backend_->Initialize()) {
                // Success!
                Logger::Info("GPUContext: Initialized backend: %s", context->backend_->GetDeviceName());
                return context;
            } else {
                Logger::Warning("GPUContext: Failed to initialize backend: %d", static_cast<int>(backend_to_try));
            }
            
            // Fallback logic
            if (backend_to_try == GPUBackendType::CUDA) {
                // CUDA failed, try Metal (if on macOS)
                backend_to_try = GPUBackendType::METAL;
            } else if (backend_to_try == GPUBackendType::METAL) {
                // Metal failed, fallback to CPU
                backend_to_try = GPUBackendType::CPU;
            } else {
                // CPU is the last resort
                backend_to_try = GPUBackendType::CPU;
                break;
            }
        }
        
        // If we reach here, create CPU fallback (always succeeds)
        context->backend_ = std::make_unique<CPUFallback>();
        context->backend_->Initialize();
        Logger::Info("GPUContext: Using CPU fallback: %s", context->backend_->GetDeviceName());

        return context;
    }

    GPUContext::~GPUContext() {
        if (backend_) {
            backend_->Shutdown();
        }
    }

    GPUBackendType GPUContext::GetBackendType() const {
        return backend_ ? backend_->GetType() : GPUBackendType::CPU;
    }

    bool GPUContext::IsGPUAccelerated() const {
        auto type = GetBackendType();
        return (type == GPUBackendType::CUDA) || (type == GPUBackendType::METAL);
    }

    void GPUContext::FallbackToCPU() {
        if (GetBackendType() == GPUBackendType::CPU) {
            return; // Already on CPU
        }
        
        // Shutdown current backend
        if (backend_) {
            backend_->Shutdown();
        }
        
        // Switch to CPU
        backend_ = std::make_unique<CPUFallback>();
        backend_->Initialize();
    }

    GPUBackendType GPUContext::DetectBestBackend() {
#ifdef CINEMATICFX_PLATFORM_WINDOWS
        // Windows: Prefer CUDA
        if (CUDABackend::IsAvailable()) {
            return GPUBackendType::CUDA;
        }
#elif defined(CINEMATICFX_PLATFORM_MACOS)
        // macOS: Prefer Metal
        if (MetalBackend::IsAvailable()) {
            return GPUBackendType::METAL;
        }
#endif
        // Default: CPU fallback
        return GPUBackendType::CPU;
    }

    std::unique_ptr<IGPUBackend> GPUContext::CreateBackend(GPUBackendType type) {
        switch (type) {
            case GPUBackendType::CUDA:
#ifdef CINEMATICFX_CUDA_AVAILABLE
                if (CUDABackend::IsAvailable()) {
                    return std::make_unique<CUDABackend>();
                }
#endif
                return nullptr;
                
            case GPUBackendType::METAL:
#ifdef CINEMATICFX_PLATFORM_MACOS
                if (MetalBackend::IsAvailable()) {
                    return std::make_unique<MetalBackend>();
                }
#endif
                return nullptr;
                
            case GPUBackendType::CPU:
                return std::make_unique<CPUFallback>();
                
            default:
                return nullptr;
        }
    }

} // namespace CinematicFX

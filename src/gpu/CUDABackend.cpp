/*******************************************************************************
 * CinematicFX - CUDA Backend Implementation
 * 
 * NVIDIA GPU acceleration for all 5 cinematic effects
 ******************************************************************************/

#include "CUDABackend.h"
#include "../utils/Logger.h"

#ifdef CINEMATICFX_CUDA_AVAILABLE
#include <cuda_runtime.h>

#include <vector>
#include <cstring>

// CUDA kernel declarations (implemented in .cu files)
extern "C" {
    void bloom_extract_cuda(const float* input, float* output, int width, int height,
                            float threshold, float shadow_lift, cudaStream_t stream);
    void bloom_blur_horizontal_cuda(const float* input, float* output, int width, int height,
                                     float radius, cudaStream_t stream);
    void bloom_blur_vertical_cuda(const float* input, float* output, int width, int height,
                                   float radius, cudaStream_t stream);
    void bloom_blend_cuda(const float* original, const float* bloom, float* output,
                          int width, int height, float amount, float r, float g, float b,
                          cudaStream_t stream);
    
    void glow_extract_cuda(const float* input, float* output, int width, int height,
                           float threshold, cudaStream_t stream);
    void glow_blur_cuda(const float* input, float* output, int width, int height,
                        float radius, cudaStream_t stream);
    void glow_blend_cuda(const float* original, const float* glow, float* output,
                         int width, int height, float intensity, cudaStream_t stream);
    
    void halation_extract_cuda(const float* input, float* output, int width, int height,
                                float threshold, cudaStream_t stream);
    void halation_blur_cuda(const float* input, float* output, int width, int height,
                            float radius, cudaStream_t stream);
    void halation_blend_cuda(const float* original, const float* halation, float* output,
                             int width, int height, float intensity, cudaStream_t stream);
    
    void grain_apply_cuda(const float* input, float* output, int width, int height,
                          float amount, float size, float roughness, int frame_number,
                          cudaStream_t stream);
    
    void chromatic_aberration_apply_cuda(const float* input, float* output, int width, int height,
                                          float amount, float angle, cudaStream_t stream);
}

#endif // CINEMATICFX_CUDA_AVAILABLE

namespace CinematicFX {

#ifdef CINEMATICFX_CUDA_AVAILABLE

struct CUDABackend::CUDAContext {
    int device_id;
    cudaStream_t stream;
    std::vector<void*> allocated_buffers;
    
    CUDAContext() : device_id(-1), stream(nullptr) {}
};

#else

// Stub context when CUDA is not available
struct CUDABackend::CUDAContext {
    int device_id;
    CUDAContext() : device_id(-1) {}
};

#endif // CINEMATICFX_CUDA_AVAILABLE

CUDABackend::CUDABackend()
    : cuda_ctx_(new CUDAContext())
{
}

CUDABackend::~CUDABackend() {
    Shutdown();
    delete cuda_ctx_;
}

bool CUDABackend::IsAvailable() {
#ifdef CINEMATICFX_CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

#ifdef CINEMATICFX_CUDA_AVAILABLE

bool CUDABackend::Initialize() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        Logger::Warning("CUDA: No CUDA devices found (%s)", cudaGetErrorString(err));
        return false;
    }
    
    // Select best device (highest compute capability)
    int best_device = 0;
    int max_compute = 0;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        int compute = prop.major * 10 + prop.minor;
        if (compute > max_compute) {
            max_compute = compute;
            best_device = i;
        }
    }
    
    cuda_ctx_->device_id = best_device;
    
    // Set device and create stream
    cudaSetDevice(cuda_ctx_->device_id);
    cudaStreamCreate(&cuda_ctx_->stream);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_ctx_->device_id);
    
    Logger::Info("CUDA: Initialized on device %d: %s (SM %d.%d)",
                 cuda_ctx_->device_id, prop.name, prop.major, prop.minor);
    Logger::Info("CUDA: Total memory: %.1f GB", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    
    return true;
}

void CUDABackend::Shutdown() {
    if (cuda_ctx_->stream) {
        cudaStreamSynchronize(cuda_ctx_->stream);
        cudaStreamDestroy(cuda_ctx_->stream);
        cuda_ctx_->stream = nullptr;
    }
    
    // Free all allocated buffers
    for (void* buffer : cuda_ctx_->allocated_buffers) {
        cudaFree(buffer);
    }
    cuda_ctx_->allocated_buffers.clear();
    
    if (cuda_ctx_->device_id >= 0) {
        cudaDeviceReset();
        cuda_ctx_->device_id = -1;
    }
    
    Logger::Info("CUDA: Shutdown complete");
}

const char* CUDABackend::GetDeviceName() const {
    if (cuda_ctx_->device_id < 0) {
        return "CUDA (Not initialized)";
    }
    
    static char device_name[256];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_ctx_->device_id);
    snprintf(device_name, sizeof(device_name), "CUDA - %s", prop.name);
    
    return device_name;
}

uint64_t CUDABackend::GetAvailableMemory() const {
    if (cuda_ctx_->device_id < 0) {
        return 0;
    }
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return static_cast<uint64_t>(free_mem);
}

GPUTexture CUDABackend::AllocateTexture(uint32_t width, uint32_t height) {
    size_t size = width * height * 4 * sizeof(float); // RGBA float
    
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size);
    
    if (err != cudaSuccess) {
        Logger::Error("CUDA: Failed to allocate texture (%ux%u): %s",
                      width, height, cudaGetErrorString(err));
        return nullptr;
    }
    
    cuda_ctx_->allocated_buffers.push_back(device_ptr);
    
    Logger::Debug("CUDA: Allocated texture %ux%u (%.2f MB)",
                  width, height, size / (1024.0f * 1024.0f));
    
    return device_ptr;
}

void CUDABackend::ReleaseTexture(GPUTexture texture) {
    if (!texture) {
        return;
    }
    
    auto it = std::find(cuda_ctx_->allocated_buffers.begin(),
                        cuda_ctx_->allocated_buffers.end(),
                        texture);
    
    if (it != cuda_ctx_->allocated_buffers.end()) {
        cudaFree(texture);
        cuda_ctx_->allocated_buffers.erase(it);
        Logger::Debug("CUDA: Released texture");
    } else {
        Logger::Warning("CUDA: Attempted to release unknown texture");
    }
}

bool CUDABackend::UploadTexture(GPUTexture dst, const FrameBuffer& src) {
    size_t size = src.width * src.height * 4 * sizeof(float);
    
    cudaError_t err = cudaMemcpyAsync(dst, src.data, size,
                                      cudaMemcpyHostToDevice,
                                      cuda_ctx_->stream);
    
    if (err != cudaSuccess) {
        Logger::Error("CUDA: Upload failed: %s", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

bool CUDABackend::DownloadTexture(const GPUTexture src, FrameBuffer& dst) {
    size_t size = dst.width * dst.height * 4 * sizeof(float);
    
    cudaError_t err = cudaMemcpyAsync(dst.data, src, size,
                                      cudaMemcpyDeviceToHost,
                                      cuda_ctx_->stream);
    
    if (err != cudaSuccess) {
        Logger::Error("CUDA: Download failed: %s", cudaGetErrorString(err));
        return false;
    }
    
    // Synchronize to ensure data is ready
    cudaStreamSynchronize(cuda_ctx_->stream);
    
    return true;
}

bool CUDABackend::ExecuteBloom(GPUTexture input, GPUTexture output, GPUTexture temp,
                               uint32_t width, uint32_t height,
                               const BloomParameters& params) {
    // Pass 1: Extract bright areas with shadow lift
    bloom_extract_cuda((const float*)input, (float*)temp,
                       width, height,
                       params.threshold, params.shadow_lift,
                       cuda_ctx_->stream);
    
    // Pass 2: Horizontal blur
    bloom_blur_horizontal_cuda((const float*)temp, (float*)output,
                               width, height,
                               params.radius,
                               cuda_ctx_->stream);
    
    // Pass 3: Vertical blur
    bloom_blur_vertical_cuda((const float*)output, (float*)temp,
                             width, height,
                             params.radius,
                             cuda_ctx_->stream);
    
    // Pass 4: Blend with original using tint
    bloom_blend_cuda((const float*)input, (const float*)temp, (float*)output,
                     width, height,
                     params.amount, params.tint_r, params.tint_g, params.tint_b,
                     cuda_ctx_->stream);
    
    return true;
}

bool CUDABackend::ExecuteGlow(GPUTexture input, GPUTexture output, GPUTexture temp,
                              uint32_t width, uint32_t height,
                              const GlowParameters& params) {
    // Pass 1: Extract highlights above threshold
    glow_extract_cuda((const float*)input, (float*)temp,
                      width, height,
                      params.threshold,
                      cuda_ctx_->stream);
    
    // Pass 2: Large blur for diffusion
    glow_blur_cuda((const float*)temp, (float*)output,
                   width, height,
                   params.diffusion_radius,
                   cuda_ctx_->stream);
    
    // Pass 3: Blend with original
    glow_blend_cuda((const float*)input, (const float*)output, (float*)temp,
                    width, height,
                    params.intensity,
                    cuda_ctx_->stream);
    
    // Copy result back to output
    cudaMemcpyAsync(output, temp,
                    width * height * 4 * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    cuda_ctx_->stream);
    
    return true;
}

bool CUDABackend::ExecuteHalation(GPUTexture input, GPUTexture output, GPUTexture temp,
                                  uint32_t width, uint32_t height,
                                  const HalationParameters& params) {
    // Pass 1: Extract extreme highlights (red channel only)
    halation_extract_cuda((const float*)input, (float*)temp,
                          width, height,
                          0.9f, // High threshold for halation
                          cuda_ctx_->stream);
    
    // Pass 2: Blur the red fringe
    halation_blur_cuda((const float*)temp, (float*)output,
                       width, height,
                       params.spread,
                       cuda_ctx_->stream);
    
    // Pass 3: Blend red fringe with original
    halation_blend_cuda((const float*)input, (const float*)output, (float*)temp,
                        width, height,
                        params.intensity,
                        cuda_ctx_->stream);
    
    // Copy result back to output
    cudaMemcpyAsync(output, temp,
                    width * height * 4 * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    cuda_ctx_->stream);
    
    return true;
}

bool CUDABackend::ExecuteGrain(GPUTexture input, GPUTexture output,
                               uint32_t width, uint32_t height,
                               const GrainParameters& params, int frame_number) {
    grain_apply_cuda((const float*)input, (float*)output,
                     width, height,
                     params.amount, params.size, params.roughness,
                     frame_number,
                     cuda_ctx_->stream);
    
    return true;
}

bool CUDABackend::ExecuteChromaticAberration(GPUTexture input, GPUTexture output,
                                             uint32_t width, uint32_t height,
                                             const ChromaticAberrationParameters& params) {
    chromatic_aberration_apply_cuda((const float*)input, (float*)output,
                                     width, height,
                                     params.amount, params.angle,
                                     cuda_ctx_->stream);
    
    return true;
}

void CUDABackend::Synchronize() {
    if (cuda_ctx_->stream) {
        cudaStreamSynchronize(cuda_ctx_->stream);
    }
}

#else // !CINEMATICFX_CUDA_AVAILABLE

// Stub implementations when CUDA is not available
bool CUDABackend::Initialize() { return false; }
void CUDABackend::Shutdown() {}
const char* CUDABackend::GetDeviceName() const { return "CUDA (unavailable)"; }
uint64_t CUDABackend::GetAvailableMemory() const { return 0; }
GPUTexture CUDABackend::UploadTexture(const FrameBuffer& buffer) { (void)buffer; return nullptr; }
bool CUDABackend::DownloadTexture(GPUTexture texture, FrameBuffer& buffer) { (void)texture; (void)buffer; return false; }
void CUDABackend::ReleaseTexture(GPUTexture texture) { (void)texture; }
GPUTexture CUDABackend::AllocateTexture(uint32_t width, uint32_t height) { (void)width; (void)height; return nullptr; }
void CUDABackend::ExecuteBloom(GPUTexture input, GPUTexture output, const BloomParameters& params) { (void)input; (void)output; (void)params; }
void CUDABackend::ExecuteGlow(GPUTexture input, GPUTexture output, const GlowParameters& params) { (void)input; (void)output; (void)params; }
void CUDABackend::ExecuteHalation(GPUTexture input, GPUTexture output, const HalationParameters& params) { (void)input; (void)output; (void)params; }
void CUDABackend::ExecuteGrain(GPUTexture input, GPUTexture output, const GrainParameters& params, uint32_t frame_number) { (void)input; (void)output; (void)params; (void)frame_number; }
void CUDABackend::ExecuteChromaticAberration(GPUTexture input, GPUTexture output, const ChromaticAberrationParameters& params) { (void)input; (void)output; (void)params; }
void CUDABackend::Synchronize() {}

#endif // CINEMATICFX_CUDA_AVAILABLE

} // namespace CinematicFX

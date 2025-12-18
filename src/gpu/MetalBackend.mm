/*******************************************************************************
 * CinematicFX - Metal Backend Implementation (macOS)
 * 
 * GPU acceleration using Apple Metal API
 ******************************************************************************/

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "MetalBackend.h"
#include "../utils/Logger.h"
#include <vector>
#include <map>

namespace CinematicFX {

// Metal texture wrapper
struct MetalTexture {
    id<MTLTexture> texture;
    uint32_t width;
    uint32_t height;
};

class MetalBackend::Impl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    std::vector<MetalTexture*> allocated_textures;
    
    bool Initialize();
    void Shutdown();
    MetalTexture* GetMetalTexture(GPUTexture handle);
};

MetalBackend::MetalBackend() 
    : impl_(new Impl())
    , initialized_(false) {
}

MetalBackend::~MetalBackend() {
    Shutdown();
    delete impl_;
}

bool MetalBackend::Initialize() {
    if (initialized_) {
        return true;
    }
    
    Logger::Info("MetalBackend: Initializing Metal GPU backend");
    
    // Get default Metal device
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        Logger::Error("MetalBackend: No Metal-capable device found");
        return false;
    }
    
    Logger::Info("MetalBackend: Using GPU: %s", [impl_->device.name UTF8String]);
    
    // Create command queue
    impl_->commandQueue = [impl_->device newCommandQueue];
    if (!impl_->commandQueue) {
        Logger::Error("MetalBackend: Failed to create command queue");
        return false;
    }
    
    // Load default shader library
    NSError* error = nil;
    impl_->library = [impl_->device newDefaultLibrary];
    if (!impl_->library) {
        Logger::Warning("MetalBackend: No default shader library, will use MPS");
    }
    
    initialized_ = true;
    Logger::Info("MetalBackend: Initialization complete");
    return true;
}

void MetalBackend::Shutdown() {
    if (!initialized_) {
        return;
    }
    
    // Release all textures
    for (auto* texture : impl_->allocated_textures) {
        delete texture;
    }
    impl_->allocated_textures.clear();
    
    // Release Metal objects
    impl_->pipelines.clear();
    impl_->library = nil;
    impl_->commandQueue = nil;
    impl_->device = nil;
    
    initialized_ = false;
    Logger::Info("MetalBackend: Shutdown complete");
}

const char* MetalBackend::GetDeviceName() const {
    if (impl_->device) {
        return [impl_->device.name UTF8String];
    }
    return "Unknown Metal Device";
}

GPUTexture MetalBackend::AllocateTexture(uint32_t width, uint32_t height) {
    if (!initialized_) {
        return nullptr;
    }
    
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor 
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
        width:width
        height:height
        mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    descriptor.storageMode = MTLStorageModePrivate;
    
    id<MTLTexture> metalTexture = [impl_->device newTextureWithDescriptor:descriptor];
    if (!metalTexture) {
        Logger::Error("MetalBackend: Failed to allocate texture");
        return nullptr;
    }
    
    auto* texture = new MetalTexture();
    texture->texture = metalTexture;
    texture->width = width;
    texture->height = height;
    
    impl_->allocated_textures.push_back(texture);
    return reinterpret_cast<GPUTexture>(texture);
}

void MetalBackend::ReleaseTexture(GPUTexture texture) {
    auto* metalTexture = impl_->GetMetalTexture(texture);
    if (!metalTexture) {
        return;
    }
    
    auto it = std::find(impl_->allocated_textures.begin(), 
                       impl_->allocated_textures.end(), 
                       metalTexture);
    if (it != impl_->allocated_textures.end()) {
        impl_->allocated_textures.erase(it);
    }
    
    delete metalTexture;
}

GPUTexture MetalBackend::UploadTexture(const FrameBuffer& buffer) {
    if (!initialized_) {
        return nullptr;
    }
    
    GPUTexture texture = AllocateTexture(buffer.width, buffer.height);
    if (!texture) {
        return nullptr;
    }
    
    auto* metalTexture = impl_->GetMetalTexture(texture);
    MTLRegion region = MTLRegionMake2D(0, 0, buffer.width, buffer.height);
    
    [metalTexture->texture replaceRegion:region
                             mipmapLevel:0
                               withBytes:buffer.data
                             bytesPerRow:buffer.stride * sizeof(float) * 4];
    
    return texture;
}

bool MetalBackend::DownloadTexture(GPUTexture texture, FrameBuffer& buffer) {
    auto* metalTexture = impl_->GetMetalTexture(texture);
    if (!metalTexture) {
        return false;
    }
    
    MTLRegion region = MTLRegionMake2D(0, 0, metalTexture->width, metalTexture->height);
    
    [metalTexture->texture getBytes:buffer.data
                        bytesPerRow:buffer.stride * sizeof(float) * 4
                         fromRegion:region
                        mipmapLevel:0];
    
    return true;
}

bool MetalBackend::ExecuteBloom(GPUTexture input, GPUTexture output, const BloomParameters& params) {
    auto* inputTex = impl_->GetMetalTexture(input);
    auto* outputTex = impl_->GetMetalTexture(output);
    
    if (!inputTex || !outputTex) {
        return false;
    }
    
    id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
    
    // Use Metal Performance Shaders for Gaussian blur
    MPSImageGaussianBlur* blur = [[MPSImageGaussianBlur alloc] 
        initWithDevice:impl_->device sigma:params.radius / 3.0f];
    
    // Create temporary texture for threshold
    GPUTexture tempTexture = AllocateTexture(inputTex->width, inputTex->height);
    auto* temp = impl_->GetMetalTexture(tempTexture);
    
    // Apply threshold and blur (simplified - would need custom shader for threshold)
    [blur encodeToCommandBuffer:commandBuffer 
                   sourceTexture:inputTex->texture
              destinationTexture:temp->texture];
    
    // Blend with original (would need custom shader)
    [blur encodeToCommandBuffer:commandBuffer 
                   sourceTexture:temp->texture
              destinationTexture:outputTex->texture];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    ReleaseTexture(tempTexture);
    return true;
}

bool MetalBackend::ExecuteGlow(GPUTexture input, GPUTexture output, const GlowParameters& params) {
    auto* inputTex = impl_->GetMetalTexture(input);
    auto* outputTex = impl_->GetMetalTexture(output);
    
    if (!inputTex || !outputTex) {
        return false;
    }
    
    id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
    
    // Multi-pass blur for glow effect
    MPSImageGaussianBlur* blur = [[MPSImageGaussianBlur alloc] 
        initWithDevice:impl_->device sigma:params.diffusion_radius / 3.0f];
    
    [blur encodeToCommandBuffer:commandBuffer 
                   sourceTexture:inputTex->texture
              destinationTexture:outputTex->texture];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return true;
}

bool MetalBackend::ExecuteHalation(GPUTexture input, GPUTexture output, const HalationParameters& params) {
    auto* inputTex = impl_->GetMetalTexture(input);
    auto* outputTex = impl_->GetMetalTexture(output);
    
    if (!inputTex || !outputTex) {
        return false;
    }
    
    id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
    
    // Large radius blur for halation
    MPSImageGaussianBlur* blur = [[MPSImageGaussianBlur alloc] 
        initWithDevice:impl_->device sigma:params.spread / 2.0f];
    
    [blur encodeToCommandBuffer:commandBuffer 
                   sourceTexture:inputTex->texture
              destinationTexture:outputTex->texture];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return true;
}

bool MetalBackend::ExecuteGrain(GPUTexture input, GPUTexture output, const GrainParameters& params, uint32_t frame) {
    auto* inputTex = impl_->GetMetalTexture(input);
    auto* outputTex = impl_->GetMetalTexture(output);
    
    if (!inputTex || !outputTex) {
        return false;
    }
    
    // For now, just copy (grain would need custom shader)
    id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    
    [blitEncoder copyFromTexture:inputTex->texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(inputTex->width, inputTex->height, 1)
                       toTexture:outputTex->texture
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return true;
}

bool MetalBackend::ExecuteChromaticAberration(GPUTexture input, GPUTexture output, const ChromaticAberrationParameters& params) {
    auto* inputTex = impl_->GetMetalTexture(input);
    auto* outputTex = impl_->GetMetalTexture(output);
    
    if (!inputTex || !outputTex) {
        return false;
    }
    
    // For now, just copy (chromatic aberration needs custom shader)
    id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    
    [blitEncoder copyFromTexture:inputTex->texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(inputTex->width, inputTex->height, 1)
                       toTexture:outputTex->texture
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return true;
}

void MetalBackend::Synchronize() {
    // Metal command buffers are synchronous by default when using waitUntilCompleted
}

MetalTexture* MetalBackend::Impl::GetMetalTexture(GPUTexture handle) {
    return reinterpret_cast<MetalTexture*>(handle);
}

} // namespace CinematicFX

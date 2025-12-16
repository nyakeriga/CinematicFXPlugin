/*******************************************************************************
 * CinematicFX - Texture Manager Implementation
 * 
 * GPU texture pool for efficient memory reuse
 ******************************************************************************/

#include "../include/GPUInterface.h"
#include "../utils/Logger.h"
#include <vector>
#include <algorithm>

namespace CinematicFX {

// Texture pool entry
struct TexturePoolEntry {
    GPUTexture texture;
    uint32_t width;
    uint32_t height;
    bool in_use;
    
    TexturePoolEntry(GPUTexture tex, uint32_t w, uint32_t h)
        : texture(tex), width(w), height(h), in_use(false) {}
};

// Internal texture pool implementation
struct TextureManager::TexturePool {
    std::vector<TexturePoolEntry> entries;
};

TextureManager::TextureManager(GPUContext* context)
    : context_(context)
    , pool_(new TexturePool())
{
    Logger::Debug("TextureManager: Created");
}

TextureManager::~TextureManager() {
    ClearPool();
    pool_.reset();
    Logger::Debug("TextureManager: Destroyed");
}

GPUTexture TextureManager::AcquireTexture(uint32_t width, uint32_t height) {
    if (!context_ || !context_->GetBackend()) {
        Logger::Error("TextureManager: No valid GPU context");
        return nullptr;
    }
    
    // Look for existing unused texture with matching dimensions
    for (auto& entry : pool_->entries) {
        if (!entry.in_use && entry.width == width && entry.height == height) {
            entry.in_use = true;
            Logger::Debug("TextureManager: Reusing texture (%ux%u)", width, height);
            return entry.texture;
        }
    }
    
    // No suitable texture found - allocate new one
    GPUTexture new_texture = context_->GetBackend()->AllocateTexture(width, height);
    if (!new_texture) {
        Logger::Error("TextureManager: Failed to allocate texture (%ux%u)", width, height);
        return nullptr;
    }
    
    pool_->entries.emplace_back(new_texture, width, height);
    pool_->entries.back().in_use = true;
    
    Logger::Debug("TextureManager: Allocated new texture (%ux%u), pool size: %zu", 
                  width, height, pool_->entries.size());
    
    return new_texture;
}

void TextureManager::ReleaseTexture(GPUTexture texture) {
    if (!texture) {
        return;
    }
    
    // Mark texture as not in use (keep in pool for reuse)
    for (auto& entry : pool_->entries) {
        if (entry.texture == texture) {
            entry.in_use = false;
            Logger::Debug("TextureManager: Released texture back to pool");
            return;
        }
    }
    
    Logger::Warning("TextureManager: Attempted to release unknown texture");
}

void TextureManager::ClearPool() {
    if (!context_ || !context_->GetBackend()) {
        return;
    }
    
    Logger::Debug("TextureManager: Clearing pool (%zu textures)", pool_->entries.size());
    
    IGPUBackend* backend = context_->GetBackend();
    
    for (auto& entry : pool_->entries) {
        backend->ReleaseTexture(entry.texture);
    }
    
    pool_->entries.clear();
}

} // namespace CinematicFX

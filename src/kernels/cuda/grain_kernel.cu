/*******************************************************************************
 * CinematicFX - CUDA Grain Kernel
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Perlin noise implementation on GPU
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

__device__ float grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__constant__ unsigned char perm_table[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__device__ float perlin3d(float x, float y, float z) {
    int X = int(floorf(x)) & 255;
    int Y = int(floorf(y)) & 255;
    int Z = int(floorf(z)) & 255;
    
    x -= floorf(x);
    y -= floorf(y);
    z -= floorf(z);
    
    float u = fade(x);
    float v = fade(y);
    float w = fade(z);
    
    int A = perm_table[X] + Y;
    int AA = perm_table[A] + Z;
    int AB = perm_table[A + 1] + Z;
    int B = perm_table[X + 1] + Y;
    int BA = perm_table[B] + Z;
    int BB = perm_table[B + 1] + Z;
    
    return lerp(w, lerp(v, lerp(u, grad(perm_table[AA], x, y, z),
                                     grad(perm_table[BA], x - 1, y, z)),
                             lerp(u, grad(perm_table[AB], x, y - 1, z),
                                     grad(perm_table[BB], x - 1, y - 1, z))),
                     lerp(v, lerp(u, grad(perm_table[AA + 1], x, y, z - 1),
                                     grad(perm_table[BA + 1], x - 1, y, z - 1)),
                             lerp(u, grad(perm_table[AB + 1], x, y - 1, z - 1),
                                     grad(perm_table[BB + 1], x - 1, y - 1, z - 1))));
}

// Apply film grain
__global__ void grain_apply_kernel(const float* input, float* output,
                                   int width, int height,
                                   float amount, float size, float roughness,
                                   int frame_number) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    
    float r = input[idx + 0];
    float g = input[idx + 1];
    float b = input[idx + 2];
    float a = input[idx + 3];
    
    // Calculate luminance for grain intensity mapping
    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    
    // Perlin noise coordinates
    float noise_x = float(x) / size;
    float noise_y = float(y) / size;
    float noise_z = float(frame_number) / 30.0f; // Temporal variation
    
    // Generate noise
    float noise = perlin3d(noise_x, noise_y, noise_z);
    
    // Remap noise from [-1,1] to [0,1]
    noise = (noise + 1.0f) * 0.5f;
    
    // Apply roughness
    noise = powf(noise, 1.0f / (roughness + 0.001f));
    
    // Remap to [-1,1]
    noise = noise * 2.0f - 1.0f;
    
    // Luminosity-based grain intensity
    float luma_factor = 1.0f - fabsf(luma - 0.5f) * 2.0f;
    luma_factor = luma_factor * luma_factor; // Curve it
    
    float grain_intensity = amount * luma_factor * noise;
    
    // Apply grain
    output[idx + 0] = fminf(fmaxf(r + grain_intensity, 0.0f), 1.0f);
    output[idx + 1] = fminf(fmaxf(g + grain_intensity, 0.0f), 1.0f);
    output[idx + 2] = fminf(fmaxf(b + grain_intensity, 0.0f), 1.0f);
    output[idx + 3] = a;
}

// Host launcher
extern "C" {

void grain_apply_cuda(const float* input, float* output, int width, int height,
                      float amount, float size, float roughness, int frame_number,
                      cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    grain_apply_kernel<<<grid, block, 0, stream>>>(input, output, width, height,
                                                     amount, size, roughness, frame_number);
}

} // extern "C"

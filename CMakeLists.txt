################################################################################
# CinematicFX Plugin - CMake Build System
# Cross-platform build for Windows (CUDA) and macOS (Metal)
################################################################################

cmake_minimum_required(VERSION 3.20)

project(CinematicFX
    VERSION 1.0.0
    DESCRIPTION "Professional Cinematic Effects Plugin for Adobe Premiere Pro"
    LANGUAGES CXX
)

# FORCE RELEASE BUILD - Premiere Pro doesn't work with Debug builds!
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
add_definitions(-DNDEBUG)

# MAXIMUM PERFORMANCE OPTIMIZATIONS
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /Ob3 /GL /Gy /arch:AVX2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:fast /Qpar /GS- /Gw /Zc:inline")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF")

# Exception handling mode for Premiere Pro compatibility
add_definitions(-D_HAS_EXCEPTIONS=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(CINEMATICFX_BUILD_CUDA "Build with CUDA support" ON)
option(CINEMATICFX_BUILD_METAL "Build with Metal support" ON)
option(CINEMATICFX_BUILD_TESTS "Build unit tests" ON)

# Platform detection
if(WIN32)
    set(CINEMATICFX_PLATFORM "Windows")
    add_definitions(-DCINEMATICFX_PLATFORM_WINDOWS)
elseif(APPLE)
    set(CINEMATICFX_PLATFORM "macOS")
    add_definitions(-DCINEMATICFX_PLATFORM_MACOS)
endif()

message(STATUS "Building CinematicFX for ${CINEMATICFX_PLATFORM}")

################################################################################
# Adobe SDK Configuration
################################################################################

# Adobe After Effects SDK path
set(ADOBE_SDK_ROOT "C:/Users/Admin/Downloads/AfterEffectsSDK_25.6_61_win/AfterEffectsSDK_25.6_61_win/ae25.6_61.64bit.AfterEffectsSDK/Examples" CACHE PATH "Path to Adobe After Effects SDK")

if(NOT ADOBE_SDK_ROOT)
    message(WARNING "ADOBE_SDK_ROOT not set. Please provide path to AE SDK.")
endif()

include_directories(
    ${ADOBE_SDK_ROOT}/Headers
    ${ADOBE_SDK_ROOT}/Headers/SP
    ${ADOBE_SDK_ROOT}/Util
)

################################################################################
# CUDA Configuration (Windows/Linux)
################################################################################

if(CINEMATICFX_BUILD_CUDA AND NOT APPLE)
    # Try to find CUDA explicitly on Windows
    # CUDA support - UPDATE VERSION NUMBER BELOW (v12.4, v12.5, v12.6, etc.)
    if(WIN32 AND EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")
    elseif(WIN32 AND EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe")
    elseif(WIN32 AND EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
    endif()
    
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        add_definitions(-DCINEMATICFX_CUDA_AVAILABLE)
        
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90) # SM 6.0 to 9.0
        
        set(CUDA_KERNEL_SOURCES
            src/kernels/cuda/bloom_kernel.cu
            src/kernels/cuda/glow_kernel.cu
            src/kernels/cuda/halation_kernel.cu
            src/kernels/cuda/grain_kernel.cu
            src/kernels/cuda/chromatic_aberration_kernel.cu
        )
    else()
        message(WARNING "CUDA compiler not found. Building without CUDA support.")
        set(CINEMATICFX_BUILD_CUDA OFF)
    endif()
endif()

################################################################################
# Metal Configuration (macOS)
################################################################################

if(CINEMATICFX_BUILD_METAL AND APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    
    if(METAL_FRAMEWORK AND METALKIT_FRAMEWORK)
        message(STATUS "Metal framework found")
        add_definitions(-DCINEMATICFX_METAL_AVAILABLE)
        
        set(METAL_SHADER_SOURCES
            src/kernels/metal/bloom_shader.metal
            src/kernels/metal/glow_shader.metal
            src/kernels/metal/halation_shader.metal
            src/kernels/metal/grain_shader.metal
            src/kernels/metal/chromatic_aberration_shader.metal
        )
        
        # Custom command to compile Metal shaders
        set(METAL_LIBRARY ${CMAKE_BINARY_DIR}/CinematicFX.metallib)
        add_custom_command(
            OUTPUT ${METAL_LIBRARY}
            COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SOURCES}
            COMMAND xcrun -sdk macosx metallib *.air -o ${METAL_LIBRARY}
            DEPENDS ${METAL_SHADER_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Compiling Metal shaders"
        )
        add_custom_target(MetalShaders DEPENDS ${METAL_LIBRARY})
    else()
        message(WARNING "Metal framework not found. Building without Metal support.")
        set(CINEMATICFX_BUILD_METAL OFF)
    endif()
endif()

################################################################################
# Source Files
################################################################################

set(CORE_SOURCES
    src/core/PluginMain.cpp
    src/core/RenderPipeline.cpp
    src/gpu/GPUContext.cpp
)

set(GPU_SOURCES
    src/gpu/TextureManager.cpp
    src/gpu/CPUFallback.cpp
    src/gpu/CUDABackend.cpp
)

# MetalBackend only on macOS
if(CINEMATICFX_BUILD_METAL AND APPLE)
    list(APPEND GPU_SOURCES src/gpu/MetalBackend.mm) # Objective-C++
endif()

set(UTIL_SOURCES
    src/utils/MathUtils.cpp
    src/utils/Logger.cpp
)

################################################################################
# Main Plugin Target
################################################################################

if(WIN32)
    # Windows: .prm plugin (without PiPL resource for now)
    add_library(CinematicFX MODULE
        ${CORE_SOURCES}
        ${UTIL_SOURCES}
        ${CUDA_KERNEL_SOURCES}
    )
    
    # Add GPU sources
    target_sources(CinematicFX PRIVATE ${GPU_SOURCES})
    
    set_target_properties(CinematicFX PROPERTIES
        SUFFIX ".prm"
        OUTPUT_NAME "CinematicFX"
        LINK_FLAGS "/DEF:${CMAKE_SOURCE_DIR}/resources/CinematicFX.def"
    )
    
elseif(APPLE)
    # macOS: .plugin bundle
    add_library(CinematicFX MODULE
        ${CORE_SOURCES}
        ${GPU_SOURCES}
        ${EFFECT_SOURCES}
        ${UTIL_SOURCES}
        ${UI_SOURCES}
        resources/PiPL.r
    )
    
    set_target_properties(CinematicFX PROPERTIES
        BUNDLE TRUE
        BUNDLE_EXTENSION "plugin"
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/resources/Info.plist
        OUTPUT_NAME "CinematicFX"
    )
    
    if(CINEMATICFX_BUILD_METAL)
        add_dependencies(CinematicFX MetalShaders)
        
        # Copy Metal library into bundle
        add_custom_command(TARGET CinematicFX POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${METAL_LIBRARY}
                $<TARGET_BUNDLE_CONTENT_DIR:CinematicFX>/Resources/
        )
    endif()
    
    target_link_libraries(CinematicFX
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
endif()

# Include directories
target_include_directories(CinematicFX PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Link CUDA libraries
if(CINEMATICFX_BUILD_CUDA)
    target_link_libraries(CinematicFX ${CUDA_LIBRARIES})
    target_include_directories(CinematicFX PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# Compiler warnings and defines
if(MSVC)
    target_compile_options(CinematicFX PRIVATE /W4)
    target_compile_definitions(CinematicFX PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
else()
    target_compile_options(CinematicFX PRIVATE -Wall -Wextra -Wpedantic)
endif()

################################################################################
# Unit Tests
################################################################################

if(CINEMATICFX_BUILD_TESTS)
    message(STATUS "Tests disabled - no test files created yet")
    # TODO: Implement test files
endif()

################################################################################
# Installation
################################################################################

if(WIN32)
    install(TARGETS CinematicFX
        RUNTIME DESTINATION "C:/Program Files/Adobe/Common/Plug-ins/7.0/MediaCore"
        LIBRARY DESTINATION "C:/Program Files/Adobe/Common/Plug-ins/7.0/MediaCore"
    )
elseif(APPLE)
    install(TARGETS CinematicFX
        BUNDLE DESTINATION "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
        LIBRARY DESTINATION "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
    )
endif()

################################################################################
# Print Configuration Summary
################################################################################

message(STATUS "")
message(STATUS "CinematicFX Configuration Summary")
message(STATUS "==================================")
message(STATUS "Version:        ${PROJECT_VERSION}")
message(STATUS "Platform:       ${CINEMATICFX_PLATFORM}")
message(STATUS "CUDA Support:   ${CINEMATICFX_BUILD_CUDA}")
message(STATUS "Metal Support:  ${CINEMATICFX_BUILD_METAL}")
message(STATUS "Build Tests:    ${CINEMATICFX_BUILD_TESTS}")
message(STATUS "C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "")

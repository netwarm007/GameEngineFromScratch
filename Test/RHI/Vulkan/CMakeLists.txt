cmake_minimum_required(VERSION 4.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(VulkanShaderRegression LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
    get_filename_component(root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    if(WIN32)
        set(OS_WINDOWS 1)
    elseif(APPLE)
        set(OS_MACOS 1)
    else()
        set(OS_LINUX 1)
    endif()
    set(HAS_VULKAN 1)
    configure_file("${root}/config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config.h")
    set(config_dir "${CMAKE_CURRENT_BINARY_DIR}")
    add_subdirectory("${root}/External/src/crossguid" "${CMAKE_CURRENT_BINARY_DIR}/crossguid")
    set(XG_LIBRARY xg)
else()
    set(root "${PROJECT_SOURCE_DIR}")
    set(config_dir "${PROJECT_SOURCE_DIR}")
endif()

find_package(Vulkan REQUIRED)
find_program(GLSL_VALIDATOR NAMES glslangValidator
    HINTS "${root}/External/${CMAKE_SYSTEM_NAME}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_program(SPIRV_CROSS NAMES spirv-cross SPIRV-Cross
    HINTS "${root}/External/${CMAKE_SYSTEM_NAME}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_program(SPIRV_VAL NAMES spirv-val HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_program(SPIRV_DIS NAMES spirv-dis HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_path(CROSSGUID_INCLUDE_DIR Guid.hpp
    HINTS "${root}/External/${CMAKE_SYSTEM_NAME}/include"
          "${root}/External/src/crossguid" REQUIRED)

include("${root}/cmake/VulkanShaders.cmake")
set(shader_dir "${CMAKE_CURRENT_BINARY_DIR}/Shaders")
myge_add_vulkan_shaders(VulkanTestShaders "${root}" "${shader_dir}" "${config_dir}")
add_custom_command(OUTPUT "${shader_dir}/ShaderAbi.comp.spv"
    COMMAND "${GLSL_VALIDATOR}" ${MYGE_VULKAN_GLSLANG_OPTIONS}
        "-I${config_dir}" "-I${root}/Framework/Common"
        -e main -o "${shader_dir}/ShaderAbi.comp.spv"
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderAbi.comp.hlsl"
    DEPENDS VulkanTestShaders "${CMAKE_CURRENT_SOURCE_DIR}/ShaderAbi.comp.hlsl"
        "${root}/Framework/Common/cbuffer.h"
    VERBATIM)
add_executable(VulkanShaderTest VulkanShaderTest.cpp "${shader_dir}/ShaderAbi.comp.spv")
target_link_libraries(VulkanShaderTest PRIVATE Vulkan::Vulkan ${XG_LIBRARY})
if(USE_ISPC)
    target_link_libraries(VulkanShaderTest PRIVATE GeomMath)
endif()
target_include_directories(VulkanShaderTest PRIVATE "${config_dir}"
    "${CROSSGUID_INCLUDE_DIR}" "${root}/Framework/Common"
    "${root}/Framework/Manager" "${root}/Framework/SceneGraph"
    "${root}/Framework/GeomMath" "${root}/Framework/Algorism" "${root}/RHI")
if(MSVC)
    target_compile_options(VulkanShaderTest PRIVATE /utf-8)
endif()

add_test(NAME TEST_VulkanShaderAbi COMMAND "${CMAKE_COMMAND}"
    "-DPROBE=$<TARGET_FILE:VulkanShaderTest>"
    "-DSHADER_DIR=${shader_dir}" "-DSHADERS=${MYGE_VULKAN_SHADER_SOURCES}"
    "-DSPIRV_CROSS=${SPIRV_CROSS}" "-DSPIRV_VAL=${SPIRV_VAL}" "-DSPIRV_DIS=${SPIRV_DIS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/ShaderAbi.cmake")
add_test(NAME TEST_VulkanCompute COMMAND VulkanShaderTest "${shader_dir}")
set_tests_properties(TEST_VulkanShaderAbi TEST_VulkanCompute PROPERTIES TIMEOUT 90)

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
find_program(SPIRV_OPT NAMES spirv-opt HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_path(CROSSGUID_INCLUDE_DIR Guid.hpp
    HINTS "${root}/External/${CMAKE_SYSTEM_NAME}/include"
          "${root}/External/src/crossguid" REQUIRED)

include("${root}/cmake/VulkanShaders.cmake")
set(shader_dir "${CMAKE_CURRENT_BINARY_DIR}/Asset/Shaders/Vulkan/Engine")
myge_add_vulkan_shaders(VulkanTestShaders "${root}" "${shader_dir}" "${config_dir}")
add_custom_command(OUTPUT "${shader_dir}/ShaderAbi.comp.spv"
    COMMAND "${GLSL_VALIDATOR}" ${MYGE_VULKAN_GLSLANG_OPTIONS}
        "-I${config_dir}" "-I${root}/Framework/Common"
        -e main -o "${shader_dir}/ShaderAbi.comp.spv"
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderAbi.comp.hlsl"
    DEPENDS VulkanTestShaders "${CMAKE_CURRENT_SOURCE_DIR}/ShaderAbi.comp.hlsl"
        "${root}/Framework/Common/cbuffer.h"
    VERBATIM)
add_custom_command(OUTPUT "${shader_dir}/basic_noopt.vert.spv"
    COMMAND "${GLSL_VALIDATOR}" ${MYGE_VULKAN_GLSLANG_OPTIONS} -Od
        "-I${config_dir}" "-I${root}/Framework/Common"
        -e basic_vert_main -o "${shader_dir}/basic_noopt.vert.spv"
        "${root}/Asset/Shaders/HLSL/basic.vert.hlsl"
    DEPENDS VulkanTestShaders "${root}/Asset/Shaders/HLSL/basic.vert.hlsl"
        "${root}/Asset/Shaders/HLSL/vsoutput.h.hlsl"
        "${config_dir}/config.h" "${root}/Framework/Common/cbuffer.h"
        "${root}/Framework/Common/root_signature.h.hlsl"
    VERBATIM)
add_executable(VulkanShaderTest VulkanShaderTest.cpp
    "${shader_dir}/ShaderAbi.comp.spv" "${shader_dir}/basic_noopt.vert.spv")
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
    "-DSHADER_DIR=${shader_dir}" "-DSHADERS=${MYGE_VULKAN_SHADER_SOURCES};basic_noopt.vert"
    "-DSPIRV_CROSS=${SPIRV_CROSS}" "-DSPIRV_VAL=${SPIRV_VAL}" "-DSPIRV_DIS=${SPIRV_DIS}"
    "-DSPIRV_OPT=${SPIRV_OPT}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/ShaderAbi.cmake")
add_test(NAME TEST_VulkanCompute COMMAND VulkanShaderTest "${shader_dir}")
set_tests_properties(TEST_VulkanShaderAbi TEST_VulkanCompute PROPERTIES TIMEOUT 90)

add_library(VulkanRegressionRHI STATIC "${root}/RHI/Vulkan/VulkanRHI.cpp"
    "${root}/RHI/Vulkan/VulkanRHIResources.cpp"
    "${root}/RHI/Vulkan/VulkanPipelineStateManager.cpp"
    "${root}/Framework/Manager/PipelineStateManager.cpp"
    "${root}/Framework/Manager/AssetLoader.cpp")
target_include_directories(VulkanRegressionRHI PUBLIC
    $<TARGET_PROPERTY:VulkanShaderTest,INCLUDE_DIRECTORIES>
    "${root}/Framework/Interface")
target_link_libraries(VulkanRegressionRHI PUBLIC Vulkan::Vulkan ${XG_LIBRARY})
if(USE_ISPC)
    target_link_libraries(VulkanRegressionRHI PUBLIC GeomMath)
endif()
if(MSVC)
    target_compile_options(VulkanRegressionRHI PUBLIC /utf-8)
endif()
add_executable(VulkanResourceTest VulkanResourceTest.cpp)
target_link_libraries(VulkanResourceTest PRIVATE VulkanRegressionRHI)
add_test(NAME TEST_VulkanResources COMMAND VulkanResourceTest)
set_tests_properties(TEST_VulkanResources PROPERTIES TIMEOUT 90)
add_executable(VulkanPipelineTest VulkanPipelineTest.cpp
    "${root}/Platform/Empty/AssetLoader.cpp")
target_link_libraries(VulkanPipelineTest PRIVATE VulkanRegressionRHI)
add_dependencies(VulkanPipelineTest VulkanTestShaders)
add_test(NAME TEST_VulkanPipelines COMMAND VulkanPipelineTest "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(TEST_VulkanPipelines PROPERTIES TIMEOUT 90)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    find_package(ZLIB REQUIRED)
    function(add_vulkan_regression_graphics)
        # Shared pass targets locate bundled ImGui relative to the engine root.
        set(PROJECT_SOURCE_DIR "${root}")
        list(APPEND CMAKE_MODULE_PATH "${root}/cmake")
        include("${root}/cmake/PlatformDependencies.cmake")
        include_directories(
            $<TARGET_PROPERTY:VulkanRegressionRHI,INTERFACE_INCLUDE_DIRECTORIES>
            "${root}/Framework/DrawPass" "${root}/Framework/DispatchPass"
            "${root}/Framework/DrawSubPass" "${root}/Framework/Parser"
            "${root}/Framework/Ability"
            "${root}/Framework/Geometries" "${root}/External/src"
            "${root}/External/src/imgui")
        if(MSVC)
            add_compile_options(/utf-8)
        endif()
        add_subdirectory("${root}/Framework/Common" "${CMAKE_CURRENT_BINARY_DIR}/Common")
        add_library(ImageDecoders INTERFACE)
        target_link_libraries(ImageDecoders INTERFACE Common ZLIB::ZLIB)
        add_subdirectory("${root}/Framework/SceneGraph" "${CMAKE_CURRENT_BINARY_DIR}/SceneGraph")
        add_subdirectory("${root}/Framework/DrawPass" "${CMAKE_CURRENT_BINARY_DIR}/DrawPass")
        add_subdirectory("${root}/Framework/DrawSubPass" "${CMAKE_CURRENT_BINARY_DIR}/DrawSubPass")
        add_subdirectory("${root}/Framework/DispatchPass" "${CMAKE_CURRENT_BINARY_DIR}/DispatchPass")
        add_library(VulkanRegressionGraphics STATIC
            "${root}/RHI/Vulkan/VulkanGraphicsManager.cpp"
            "${root}/Framework/Manager/GraphicsManager.cpp"
            "${root}/Framework/Manager/SceneCamera.cpp"
            "${root}/Framework/Manager/SceneLighting.cpp")
        target_link_libraries(VulkanRegressionGraphics PUBLIC VulkanRegressionRHI
            PRIVATE SceneGraph DrawPass DrawSubPass DispatchPass)
    endfunction()
    add_vulkan_regression_graphics()
else()
    add_library(VulkanRegressionGraphics STATIC "${root}/RHI/Vulkan/VulkanGraphicsManager.cpp")
    target_link_libraries(VulkanRegressionGraphics PUBLIC VulkanRegressionRHI PRIVATE RenderServices)
endif()
target_sources(VulkanRegressionGraphics PRIVATE
    "${root}/RHI/Vulkan/VulkanGraphicsGui.cpp"
    "${root}/RHI/Vulkan/VulkanGuiRenderer.cpp"
    "${root}/External/src/imgui/backends/imgui_impl_vulkan.cpp"
    "${root}/RHI/Vulkan/VulkanGraphicsRendering.cpp"
    "${root}/RHI/Vulkan/VulkanGraphicsShadow.cpp")
set_source_files_properties("${root}/External/src/imgui/backends/imgui_impl_vulkan.cpp"
    PROPERTIES COMPILE_DEFINITIONS IMGUI_IMPL_VULKAN_NO_PROTOTYPES)
target_include_directories(VulkanRegressionGraphics PUBLIC "${root}/Framework/Geometries")
target_include_directories(VulkanRegressionGraphics PUBLIC "${root}/External/src/imgui")
add_executable(VulkanGraphicsTest VulkanGraphicsTest.cpp
    "${root}/Platform/Empty/AssetLoader.cpp")
target_include_directories(VulkanGraphicsTest PRIVATE "${root}/Framework/DispatchPass")
target_link_libraries(VulkanGraphicsTest PRIVATE VulkanRegressionGraphics)
add_dependencies(VulkanGraphicsTest VulkanTestShaders)
add_test(NAME TEST_VulkanGraphics COMMAND VulkanGraphicsTest "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(TEST_VulkanGraphics PROPERTIES TIMEOUT 90)
add_executable(VulkanGuiTest VulkanGuiTest.cpp "${root}/Platform/Empty/AssetLoader.cpp")
target_link_libraries(VulkanGuiTest PRIVATE VulkanRegressionGraphics)
add_test(NAME TEST_VulkanGui COMMAND VulkanGuiTest)
set_tests_properties(TEST_VulkanGui PROPERTIES TIMEOUT 90)

if(WIN32 OR UNIX)
    add_executable(VulkanFrameLoopTest VulkanFrameLoopTest.cpp "${root}/Platform/Empty/AssetLoader.cpp")
    target_link_libraries(VulkanFrameLoopTest PRIVATE VulkanRegressionGraphics)
    if(WIN32)
        target_link_libraries(VulkanFrameLoopTest PRIVATE user32)
        if(MSVC)
            target_compile_options(VulkanFrameLoopTest PRIVATE /utf-8)
        endif()
    elseif(APPLE)
        enable_language(OBJCXX)
        set_source_files_properties(VulkanFrameLoopTest.cpp PROPERTIES LANGUAGE OBJCXX)
        set_target_properties(VulkanFrameLoopTest PROPERTIES
            OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
        target_link_libraries(VulkanFrameLoopTest PRIVATE "-framework Cocoa" "-framework QuartzCore")
    else()
        find_path(XCB_INCLUDE_DIR xcb/xcb.h REQUIRED)
        find_library(XCB_LIBRARY xcb REQUIRED)
        target_include_directories(VulkanFrameLoopTest PRIVATE "${XCB_INCLUDE_DIR}")
        target_link_libraries(VulkanFrameLoopTest PRIVATE "${XCB_LIBRARY}")
    endif()
    add_test(NAME TEST_VulkanFrameLoop COMMAND VulkanFrameLoopTest)
    set_tests_properties(TEST_VulkanFrameLoop PROPERTIES TIMEOUT 90)
endif()

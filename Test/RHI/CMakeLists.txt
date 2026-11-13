if(NOT ANDROID AND NOT WA)
    if(WIN32)
        set(SHADER_TOOL_PLATFORM "Windows")
        set(SHADER_TOOL_SUFFIX ".exe")
        set(SPIRV_CROSS_NAME "SPIRV-Cross")
    else()
        set(SHADER_TOOL_PLATFORM "${CMAKE_HOST_SYSTEM_NAME}")
        set(SHADER_TOOL_SUFFIX "")
        set(SPIRV_CROSS_NAME "spirv-cross")
    endif()
    add_test(NAME TEST_ShadowShaderCompatibility
        COMMAND ${CMAKE_COMMAND}
            "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DGLSL_VALIDATOR=${PROJECT_SOURCE_DIR}/External/${SHADER_TOOL_PLATFORM}/bin/glslangValidator${SHADER_TOOL_SUFFIX}"
            "-DSPIRV_CROSS=${PROJECT_SOURCE_DIR}/External/${SHADER_TOOL_PLATFORM}/bin/${SPIRV_CROSS_NAME}${SHADER_TOOL_SUFFIX}"
            "-DAPPLE_PLATFORM=${APPLE}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/ShadowShaderCompatibility.cmake")
endif()

if(OS_MACOS OR OS_WINDOWS)
    if(Vulkan_FOUND)
        execute_process(COMMAND git submodule update --init ${MYGE_EXTERNAL_ROOT}/../src/tinyobjloader)
        add_executable(VulkanRHITest VulkanRHITest.cpp)
        target_include_directories(VulkanRHITest PRIVATE ${PROJECT_SOURCE_DIR}/External/src/tinyobjloader)
        target_link_libraries(VulkanRHITest Framework VulkanRHI PlatformInterface)
    endif(Vulkan_FOUND)
endif(OS_MACOS OR OS_WINDOWS)

if(OS_WINDOWS)
    add_executable(D2dRHITest D2dRHITest.cpp)
    target_link_libraries(D2dRHITest Framework D2dRHI PlatformInterface)

    execute_process(COMMAND git submodule update --init ${MYGE_EXTERNAL_ROOT}/../src/tinyobjloader)
    add_executable(D3d12RHITest D3d12RHITest.cpp)
    target_include_directories(D3d12RHITest PRIVATE ${PROJECT_SOURCE_DIR}/External/src/tinyobjloader)
    target_link_libraries(D3d12RHITest Framework D3dRHI PlatformInterface)

    add_executable(D3d12ComputeTest D3d12ComputeTest.cpp)
    target_link_libraries(D3d12ComputeTest Framework D3dRHI PlatformInterface)
    add_dependencies(D3d12ComputeTest Engine_Asset_Shaders)
    add_custom_command(TARGET D3d12ComputeTest PRE_BUILD
        COMMAND dxc.exe -T cs_6_1 -E main
            -I "${PROJECT_SOURCE_DIR}" -I "${PROJECT_SOURCE_DIR}/Framework/Common"
            -I "${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL"
            -Fo "${CMAKE_CURRENT_BINARY_DIR}/ShadowCoordinatesTest.cso"
            "${CMAKE_CURRENT_SOURCE_DIR}/ShadowCoordinatesTest.comp.hlsl"
        VERBATIM)
    add_test(NAME TEST_D3d12Compute COMMAND D3d12ComputeTest
        "${PROJECT_BINARY_DIR}/Asset/Shaders/HLSL/integrateBRDF.comp.cso"
        "${PROJECT_BINARY_DIR}/Asset/Shaders/HLSL/generate_mips.comp.cso"
        "${CMAKE_CURRENT_BINARY_DIR}/ShadowCoordinatesTest.cso")
endif(OS_WINDOWS)

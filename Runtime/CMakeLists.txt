# Owned startup composition shared by the desktop applications (M2.3).
# Each backend factory lives in its own translation unit and includes only its
# own backend headers, so no configuration aliases conflict. The central
# dispatch references every included factory, retaining its implementation.
set(RHI_COMPOSITION_SOURCES RhiComposition.cpp RhiBackendFactory.cpp RhiSwitch.cpp RhiFrameDriver.cpp)
if(OS_WINDOWS)
    list(APPEND RHI_COMPOSITION_SOURCES RhiCompositionD3d12.cpp)
endif()
if(OS_WINDOWS OR OS_LINUX)
    list(APPEND RHI_COMPOSITION_SOURCES RhiCompositionOpenGL.cpp)
endif()
if(OS_MACOS)
    list(APPEND RHI_COMPOSITION_SOURCES RhiCompositionMetal.cpp)
endif()
if(HAS_VULKAN)
    list(APPEND RHI_COMPOSITION_SOURCES RhiCompositionVulkan.cpp RhiCompositionVulkanProbe.cpp)
endif()

add_library(RuntimeComposition ${RHI_COMPOSITION_SOURCES})
target_include_directories(RuntimeComposition PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(RuntimeComposition PUBLIC Manager)
if(OS_LINUX AND SDL2_FOUND)
    # Keep the SDL OpenGL host (full ImGui platform integration) and take the
    # XCB host pieces for Vulkan only; the XCB OpenGLApplication and its asset
    # implementation would clash with the SDL variants in one executable.
    target_link_libraries(RuntimeComposition PUBLIC PlatformInterfaceSDL2)
    if(HAS_VULKAN)
        target_link_libraries(RuntimeComposition PUBLIC PlatformVulkanXcb)
    endif()
else()
    target_link_libraries(RuntimeComposition PUBLIC PlatformInterface)
endif()
if(OS_WINDOWS)
    target_link_libraries(RuntimeComposition PUBLIC D3dRHI)
endif()
if(OS_WINDOWS OR OS_LINUX)
    target_link_libraries(RuntimeComposition PUBLIC OpenGLRHI)
endif()
if(OS_MACOS)
    target_link_libraries(RuntimeComposition PUBLIC MetalRHI)
endif()
if(HAS_VULKAN)
    target_link_libraries(RuntimeComposition PUBLIC VulkanRHI)
endif()
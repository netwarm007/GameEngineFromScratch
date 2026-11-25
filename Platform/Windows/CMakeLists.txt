add_platform_asset_io(PlatformAssetIO AssetLoader.cpp)

set (PlatformSources 
                WindowsApplication.cpp
                Win32OpenGLBackend.cpp
                D2dApplication.cpp
                D3d12Backend.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
                imgui_impl_win32.cpp)

if(Vulkan_FOUND)
list (APPEND PlatformSources VulkanApplication.cpp)
endif(Vulkan_FOUND)

add_library(PlatformInterface
    ${PlatformSources}
)
target_link_libraries(PlatformInterface PUBLIC PlatformAssetIO)
target_link_libraries(PlatformInterface PRIVATE dwmapi)
if(TARGET VulkanLoader)
    target_link_libraries(PlatformInterface PRIVATE VulkanLoader)
endif()
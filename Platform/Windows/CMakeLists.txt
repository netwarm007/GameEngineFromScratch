set (PlatformSources 
                WindowsApplication.cpp
                OpenGLApplication.cpp
                D2dApplication.cpp
                D3d12Application.cpp
                AssetLoader.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
                imgui_impl_win32.cpp)

if(Vulkan_FOUND)
list (APPEND PlatformSources VulkanApplication.cpp)
endif(Vulkan_FOUND)

add_library(PlatformInterface
    ${PlatformSources}
)
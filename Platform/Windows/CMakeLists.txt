add_library(PlatformInterface
                WindowsApplication.cpp
                OpenGLApplication.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
                imgui_impl_win32.cpp
)

if(Vulkan_FOUND)
list (APPEND PlatformSources VulkanApplication.mm)
endif(Vulkan_FOUND)

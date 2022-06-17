add_library(PlatformInterface
                WindowsApplication.cpp
                OpenGLApplication.cpp
                VulkanApplication.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
                imgui_impl_win32.cpp
)

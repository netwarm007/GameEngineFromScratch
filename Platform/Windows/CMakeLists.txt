add_library(PlatformInterface
                WindowsApplication.cpp
                OpenGLApplication.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
                ${PROJECT_SOURCE_DIR}/External/src/imgui/examples/imgui_impl_win32.cpp
)

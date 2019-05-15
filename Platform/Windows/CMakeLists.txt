add_library(PlatformInterface
                WindowsApplication.cpp
                D3d12Application.cpp
                OpenGLApplication.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
)
target_include_directories(PlatformInterface PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")

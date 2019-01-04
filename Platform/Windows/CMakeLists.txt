find_package(OpenGL REQUIRED)

add_library(PlatformInterface
                WindowsApplication.cpp
                D3d12Application.cpp
                OpenGLApplication.cpp
                CefApplication.cpp
                simple_handler.win.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
)
target_link_libraries(PlatformInterface
                d2d1
                d3d12
                dxgi
                ${OPENGL_LIBRARIES}
)
target_include_directories(PlatformInterface PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")

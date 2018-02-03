find_package(OpenGL REQUIRED)

add_library(PlatformInterface
                WindowsApplication.cpp
                D3d12Application.cpp
                OpenGLApplication.cpp
                ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_wgl.c
)
target_link_libraries(PlatformInterface
                d2d1
                d3d12
                dxgi
                ${OPENGL_LIBRARIES}
)
target_include_directories(PlatformInterface PRIVATE 
        "${PROJECT_SOURCE_DIR}/External/src/glad/include")

add_library(MyGameEngineD3d12 
                   D3d12ApplicationConfig.cpp
)
target_link_libraries(MyGameEngineD3d12 
                        Common 
                        D3dRHI 
                        PlatformInterface
)

add_library(MyGameEngine
                OpenGLApplicationConfig.cpp
)
target_link_libraries(MyGameEngine
                        Common 
                        OpenGLRHI 
                        PlatformInterface
)
target_include_directories(MyGameEngine PRIVATE 
        "${PROJECT_SOURCE_DIR}/External/src/glad/include")



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

add_executable(MyGameEngineD3d12 
                   D3d12ApplicationConfig.cpp
)
target_link_libraries(MyGameEngineD3d12 
                        Common 
                        D3dRHI 
                        PlatformInterface
)
__add_xg_platform_dependencies(MyGameEngineD3d12)

add_custom_command(TARGET MyGameEngineD3d12 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:MyGameEngineD3d12>)

add_executable(MyGameEngineOpenGL 
                OpenGLApplicationConfig.cpp
)
target_link_libraries(MyGameEngineOpenGL 
                        Common 
                        OpenGLRHI 
                        PlatformInterface
)
target_include_directories(MyGameEngineOpenGL PRIVATE 
        "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineOpenGL)

add_custom_command(TARGET MyGameEngineOpenGL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:MyGameEngineOpenGL>)

add_subdirectory(test)


add_library(WindowsApplication WindowsApplication.cpp)

add_executable(MyGameEngineD3d12 D3d12Application.cpp)
target_link_libraries(MyGameEngineD3d12 WindowsApplication Common D3dRHI d3d12 dxgi d3dcompiler)

find_package(OpenGL REQUIRED)

add_executable(MyGameEngineOpenGL 
    OpenGLApplication.cpp
    ${PROJECT_SOURCE_DIR}/External/GL/src/glad_wgl.c
    )
target_link_libraries(MyGameEngineOpenGL WindowsApplication Common OpenGLRHI ${OPENGL_LIBRARIES})
target_include_directories(MyGameEngineOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/GL/include")

add_subdirectory(test)


add_executable(MyGameEngineD3d12 WindowsApplication.cpp D3d12Application.cpp)
target_link_libraries(MyGameEngineD3d12 Common D3dRHI d3d12 dxgi d3dcompiler)

add_executable(MyGameEngineOpenGL 
    WindowsApplication.cpp 
    OpenGLApplication.cpp
    ${PROJECT_SOURCE_DIR}/External/GL/src/glad_wgl.c
    )
target_link_libraries(MyGameEngineOpenGL Common OpenGLRHI opengl32)
target_include_directories(MyGameEngineOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/GL/include")

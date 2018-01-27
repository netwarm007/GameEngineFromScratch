find_library(COCOA_LIBRARY Cocoa required)
find_library(OPENGL_LIBRARY OpenGL required)
find_library(CV_LIBRARY CoreVideo required)

# Platform Interface library
add_library(PlatformInterface
        CocoaApplication.mm 
        CocoaOpenGLApplication.mm
        AppDelegate.m
        WindowDelegate.m
        GLView.mm
    )
target_link_libraries(PlatformInterface
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        dl
    )

# MyGameEngineOpenGL
add_library(MyGameEngine
        CocoaOpenGLApplicationConfig.cpp
    )
target_link_libraries(MyGameEngine
        Common 
        PlatformInterface
        OpenGLRHI 
        )
target_include_directories(MyGameEngine PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")


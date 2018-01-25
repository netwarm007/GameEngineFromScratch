include_directories("/opt/local/include")
link_directories("/opt/local/lib")
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

# MyGameEngineCocoa
add_executable(MyGameEngineCocoa MACOSX_BUNDLE
        CocoaApplicationConfig.cpp 
    )
target_link_libraries(MyGameEngineCocoa
        Common 
        PlatformInterface
        )
__add_xg_platform_dependencies(MyGameEngineCocoa)

# MyGameEngineCocoaOpenGL
add_executable(MyGameEngineCocoaOpenGL MACOSX_BUNDLE
        CocoaOpenGLApplicationConfig.cpp
    )
target_link_libraries(MyGameEngineCocoaOpenGL 
        Common 
        PlatformInterface
        OpenGLRHI 
        )
target_include_directories(MyGameEngineCocoaOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineCocoaOpenGL)


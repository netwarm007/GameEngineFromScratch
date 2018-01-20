include_directories("/opt/local/include")
link_directories("/opt/local/lib")
find_library(COCOA_LIBRARY Cocoa required)
find_library(OPENGL_LIBRARY OpenGL required)
find_library(CV_LIBRARY CoreVideo required)

# MyGameEngineCocoa
add_executable(MyGameEngineCocoa MACOSX_BUNDLE
        CocoaApplication.mm 
        CocoaApplicationConfig.cpp 
        AppDelegate.m
        WindowDelegate.m
    )
target_link_libraries(MyGameEngineCocoa
        Common 
        ${COCOA_LIBRARY} 
        )
__add_xg_platform_dependencies(MyGameEngineCocoa)

# MyGameEngineCocoaOpenGL
add_executable(MyGameEngineCocoaOpenGL MACOSX_BUNDLE
        CocoaApplication.mm 
        CocoaOpenGLApplication.mm
        CocoaOpenGLApplicationConfig.cpp
        AppDelegate.m
        WindowDelegate.m
        GLView.mm
    )
target_link_libraries(MyGameEngineCocoaOpenGL 
        OpenGLRHI 
        Common 
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        dl
        )
target_include_directories(MyGameEngineCocoaOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineCocoaOpenGL)


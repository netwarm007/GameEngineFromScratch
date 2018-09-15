find_library(COCOA_LIBRARY Cocoa required)
find_library(OPENGL_LIBRARY OpenGL required)
find_library(CV_LIBRARY CoreVideo required)
find_library(MTK_LIBRARY MetalKit required)

# Platform Interface library
add_library(PlatformInterface
        CocoaApplication.mm 
        CocoaOpenGLApplication.mm
        CocoaMetalApplication.mm
        AppDelegate.m
        WindowDelegate.m
        GLView.mm
        MetalView.mm
    )
target_link_libraries(PlatformInterface
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        ${MTK_LIBRARY}
        dl
    )
target_include_directories(PlatformInterface PUBLIC GeomMath)
target_link_libraries(PlatformInterface GeomMath)

# MyGameEngineOpenGL
add_library(MyGameEngineOpenGL
        CocoaOpenGLApplicationConfig.cpp
    )
target_link_libraries(MyGameEngineOpenGL
        Common 
        PlatformInterface
        OpenGLRHI 
        )
target_include_directories(MyGameEngineOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")

# MyGameEngineMetal
add_library(MyGameEngineMetal
        CocoaMetalApplicationConfig.cpp
    )
target_link_libraries(MyGameEngineMetal
        Common 
        PlatformInterface
        MetalRHI 
        )


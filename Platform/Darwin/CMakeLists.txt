include_directories("/opt/local/include")
link_directories("/opt/local/lib")
find_library(COCOA_LIBRARY Cocoa required)
find_library(OPENGL_LIBRARY OpenGL required)
find_library(CV_LIBRARY CoreVideo required)

# MyGameEngineXcbOpenGL
add_executable(MyGameEngineXcbOpenGL 
        XcbApplication.cpp 
        XcbOpenGLApplication.cpp
	    ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_glx.c
    )
target_link_libraries(MyGameEngineXcbOpenGL 
        OpenGLRHI 
        Common 
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
        xcb 
        X11 
        X11-xcb 
        dl
        )
target_include_directories(MyGameEngineXcbOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineXcbOpenGL)

# MyGameEngineCocoa
add_executable(MyGameEngineCocoa MACOSX_BUNDLE
        CocoaApplication.mm 
        CocoaApplicationConfig.cpp 
        AppDelegate.m
        WindowDelegate.m
    )
target_link_libraries(MyGameEngineCocoa
        Common 
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
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
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        dl
        )
__add_xg_platform_dependencies(MyGameEngineCocoaOpenGL)


include_directories("/opt/local/include")
link_directories("/opt/local/lib")

# MyGameEngineXcbOpenGL
add_executable(MyGameEngineXcbOpenGL XcbApplication.cpp XcbOpenGLApplication.cpp
	${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_glx.c
    )
target_link_libraries(MyGameEngineXcbOpenGL OpenGLRHI Common ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY} xcb X11 X11-xcb dl)
target_include_directories(MyGameEngineXcbOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineXcbOpenGL)

# MyGameEngineCocoaOpenGL
add_executable(MyGameEngineCocoaOpenGL MACOSX_BUNDLE
        CocoaApplication.mm 
        CocoaOpenGLApplication.mm
        AppDelegate.m
        WindowDelegate.m
        GLView.m
    )
find_library(COCOA_LIBRARY Cocoa required)
find_library(OPENGL_LIBRARY OpenGL required)
target_link_libraries(MyGameEngineCocoaOpenGL OpenGLRHI Common ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY} ${COCOA_LIBRARY} ${OPENGL_LIBRARY})
__add_xg_platform_dependencies(MyGameEngineCocoaOpenGL)



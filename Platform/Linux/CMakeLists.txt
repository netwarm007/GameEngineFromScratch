add_executable(MyGameEngineOpenGL XcbApplication.cpp OpenGLApplication.cpp
	${PROJECT_SOURCE_DIR}/External/GL/src/glad_glx.c
    )
target_link_libraries(MyGameEngineOpenGL OpenGLRHI Common xcb X11 X11-xcb dl)
target_include_directories(MyGameEngineOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/GL/include")

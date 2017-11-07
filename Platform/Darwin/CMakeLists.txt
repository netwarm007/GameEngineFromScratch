include_directories("/opt/local/include")
link_directories("/opt/local/lib")
add_executable(MyGameEngineOpenGL XcbApplication.cpp OpenGLApplication.cpp
	${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_glx.c
    )
target_link_libraries(MyGameEngineOpenGL OpenGLRHI Common ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY} xcb X11 X11-xcb dl)
target_include_directories(MyGameEngineOpenGL PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")
__add_xg_platform_dependencies(MyGameEngineOpenGL)


# Empty
add_subdirectory(Empty)
IF(OS_WINDOWS)
    # Windows specific code
    add_subdirectory(D2d)
    add_subdirectory(D3d)
    add_subdirectory(OpenGL)
ENDIF(OS_WINDOWS)
IF(APPLE)
    # Darwin specific code
    add_subdirectory(Metal)
ENDIF(APPLE)
IF(NOT OS_IOS)
add_subdirectory(OpenGL)
ENDIF(NOT OS_IOS)

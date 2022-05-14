# Empty
add_subdirectory(Empty)
IF(${WIN32})
    # Windows specific code
    add_subdirectory(D2d)
    add_subdirectory(D3d)
    add_subdirectory(OpenGL)
ENDIF(${WIN32})
IF(${APPLE})
    # Darwin specific code
    add_subdirectory(Metal)
ENDIF(${APPLE})
IF(${LINUX})
    add_subdirectory(OpenGL)
ENDIF(${LINUX})

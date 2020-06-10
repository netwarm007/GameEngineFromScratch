# Empty
add_subdirectory(Empty)
IF(${WIN32})
    # Windows specific code
    add_subdirectory(D2d)
    add_subdirectory(D3d)
ENDIF(${WIN32})
IF(${APPLE})
    # Darwin specific code
    add_subdirectory(Metal)
ENDIF(${APPLE})
add_subdirectory(OpenGL)

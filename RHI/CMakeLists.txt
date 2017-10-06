# Empty
add_subdirectory(Empty)
add_subdirectory(OpenGL)
IF(${WIN32})
    # Windows specific code
    add_subdirectory(D2d)
    add_subdirectory(D3d)
ENDIF(${WIN32})

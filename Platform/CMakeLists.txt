# Empty
add_subdirectory(Empty)
IF(${WIN32})
    # Windows specific code
    add_subdirectory(Windows)
ENDIF(${WIN32})

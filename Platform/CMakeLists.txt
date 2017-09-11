# Empty
add_subdirectory(Empty)
IF(${WIN32})
    # Windows specific code
    add_subdirectory(Windows)
ELSEIF(${UNIX})
    # Linux specific code
    add_subdirectory(Linux)
ENDIF(${WIN32})

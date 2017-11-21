# Empty
add_subdirectory(Empty)
IF(${WIN32})
    # Windows specific code
    add_subdirectory(Windows)
ELSEIF(${UNIX})
    IF(${APPLE})
        # Apple specific code
        add_subdirectory(Darwin)
    ELSE(${APPLE})
        # Linux specific code
        add_subdirectory(Linux)
    ENDIF(${APPLE})
ELSEIF(${ORBIS})
    add_subdirectory(Orbis)
ENDIF(${WIN32})

IF(${WIN32})
    # Windows specific code
    add_subdirectory(Windows)
ELSEIF(${UNIX})
    IF(${APPLE})
        # Apple specific code
        add_subdirectory(Darwin)
    ELSEIF(${ANDROID})
        add_subdirectory(Android)
    ELSE(${APPLE})
        # Linux specific code
        add_subdirectory(Linux)
    ENDIF(${APPLE})
ELSEIF(${ORBIS})
    add_subdirectory(Orbis)
ELSE()
    # Empty
    add_subdirectory(Empty)
ENDIF(${WIN32})

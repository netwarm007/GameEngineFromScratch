IF(WIN32)
    # Windows specific code
    add_subdirectory(Windows)
ELSEIF(UNIX)
    IF(APPLE)
        # Apple specific code
        add_subdirectory(Darwin)
    ELSEIF(ANDROID)
        add_subdirectory(Android)
    ELSEIF(BSD)
        add_subdirectory(Linux)
	    # CEF not support BSD yet
    ELSE(APPLE)
        # Linux specific code
        add_subdirectory(Linux)
    ENDIF(APPLE)
ELSE()
    # Empty
    add_subdirectory(Empty)
ENDIF(WIN32)

IF(WA)
    add_subdirectory(Sdl)
ENDIF(WA)

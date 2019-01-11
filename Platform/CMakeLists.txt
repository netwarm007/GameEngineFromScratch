IF(WIN32)
    # Windows specific code
    add_subdirectory(Windows)
    add_subdirectory(Cef)
ELSEIF(UNIX)
    IF(APPLE)
        # Apple specific code
        add_subdirectory(Darwin)
        add_subdirectory(Cef)
    ELSEIF(ANDROID)
        add_subdirectory(Android)
    ELSEIF(WA)
        add_subdirectory(Empty)
    ELSEIF(BSD)
        add_subdirectory(Linux)
	    # CEF not support BSD yet
    ELSE(APPLE)
        # Linux specific code
        add_subdirectory(Linux)
        add_subdirectory(Cef)
    ENDIF(APPLE)
ELSE()
    # Empty
    add_subdirectory(Empty)
ENDIF(WIN32)

add_subdirectory(Sdl)

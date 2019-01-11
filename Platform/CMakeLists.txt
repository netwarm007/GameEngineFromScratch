IF(WIN32)
    # Windows specific code
    add_subdirectory(Windows)
    add_subdirectory(Cef)
    add_subdirectory(Sdl)
ELSEIF(UNIX)
    IF(APPLE)
        # Apple specific code
        add_subdirectory(Darwin)
        add_subdirectory(Cef)
        add_subdirectory(Sdl)
    ELSEIF(ANDROID)
        add_subdirectory(Android)
        add_subdirectory(Sdl)
    ELSEIF(WA)
        add_subdirectory(Empty)
    ELSEIF(BSD)
        add_subdirectory(Linux)
        add_subdirectory(Sdl)
	# CEF not support BSD yet
    ELSE(APPLE)
        # Linux specific code
        add_subdirectory(Linux)
        add_subdirectory(Cef)
        add_subdirectory(Sdl)
    ENDIF(APPLE)
ELSE()
    # Empty
    add_subdirectory(Empty)
ENDIF(WIN32)


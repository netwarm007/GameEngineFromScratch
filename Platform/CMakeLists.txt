IF(WIN32)
    # Windows specific code
    add_subdirectory(Windows)
ELSEIF(UNIX)
    IF(OS_MACOS)
        # macOS specific code
        add_subdirectory(Darwin)
    ELSEIF(OS_IOS)
        # iOS specific code
        add_subdirectory(iOS)
    ELSEIF(ANDROID)
        add_subdirectory(Android)
    ELSEIF(WA)
        add_subdirectory(Empty)
    ELSE()
        # Linux specific code
        add_subdirectory(Linux)
    ENDIF(OS_MACOS)
ELSE()
    # Empty
    add_subdirectory(Empty)
ENDIF(WIN32)

IF(WA)
    add_subdirectory(Sdl)
ENDIF(WA)

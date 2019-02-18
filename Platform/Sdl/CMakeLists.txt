add_library(Sdl
    SdlApplication.cpp
    OpenGLApplication.cpp
)

IF(WA)
    # nothing here
ELSE(WA)
    IF(APPLE)
        list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/SDL2.framework/Resources")
    ELSEIF(UNIX)
        list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/lib/cmake/SDL2")
    ENDIF(APPLE)

    include(SDL2Config)

    target_link_libraries(Sdl
        SDL2::SDL2main
        SDL2::SDL2
    )

ENDIF(WA)

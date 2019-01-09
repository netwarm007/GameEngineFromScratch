IF(APPLE)
list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/SDL2.framework/Resources")
ENDIF(APPLE)

include(SDL2Config)

add_library(Sdl
    SdlApplication.cpp
    OpenGLApplication.cpp
)

target_link_libraries(Sdl
    SDL2::SDL2main
    SDL2::SDL2
)
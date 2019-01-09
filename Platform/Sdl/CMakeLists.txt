include(SDL2Config)

add_library(Sdl
    SdlApplication.cpp
    OpenGLApplication.cpp
)

target_link_libraries(Sdl
    SDL2::SDL2main
    SDL2::SDL2
)
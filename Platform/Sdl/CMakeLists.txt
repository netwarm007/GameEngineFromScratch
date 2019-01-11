add_library(Sdl
    SdlApplication.cpp
    OpenGLApplication.cpp
)

IF(NOT WA)
target_link_libraries(Sdl
    ${SDL2_LIBRARY}
    ${SDL2_MAIN_LIBRARY}
)
ENDIF(NOT WA)

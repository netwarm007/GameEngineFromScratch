add_library(Sdl
    SdlApplication.cpp
    OpenGLApplication.cpp
)

IF(APPLE)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/SDL2.framework/Resources")
ELSEIF(UNIX)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/lib/cmake/SDL2")
ENDIF(APPLE)

target_link_libraries(PlatformInterface
    ${SDL2_LIBRARY}
)

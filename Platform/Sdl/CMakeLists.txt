add_library(Sdl
    SdlApplication.cpp
    OpenGLApplication.cpp
)

find_package(SDL2)
include_directories(${SDL2_INCLUDE_DIRS})

IF(APPLE)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/SDL2.framework/Resources")
ELSEIF(UNIX)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/lib/cmake/SDL2")
ENDIF(APPLE)

target_link_libraries(Sdl
    ${SDL2_LIBRARIES}
)

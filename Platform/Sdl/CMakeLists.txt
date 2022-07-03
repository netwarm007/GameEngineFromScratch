add_library(PlatformInterfaceSDL2
    SdlApplication.cpp
    OpenGLApplication.cpp
    AssetLoader.cpp
    imgui_impl_sdl.cpp
)

IF(APPLE)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/SDL2.framework/Resources")
ELSEIF(UNIX)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/lib/cmake/SDL2")
ENDIF(APPLE)

target_include_directories(PlatformInterfaceSDL2 PRIVATE
    ${SDL2_INCLUDE_DIR}
)

target_link_libraries(PlatformInterfaceSDL2
    ${SDL2_LIBRARY}
)

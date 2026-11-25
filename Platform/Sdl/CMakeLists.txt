add_platform_asset_io(SdlAssetIO AssetLoader.cpp)

IF(APPLE)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/SDL2.framework/Resources")
ELSEIF(UNIX)
    list(APPEND CMAKE_MODULE_PATH "${MYGE_EXTERNAL_ROOT}/lib/cmake/SDL2")
ENDIF(APPLE)

# Shared ImGui SDL2 platform backend: linked by both the SDL host library
# and the XCB Vulkan pieces (the dual-mode Linux host uses SDL events).
add_library(ImGuiSDL2Backend imgui_impl_sdl2.cpp)
target_include_directories(ImGuiSDL2Backend PUBLIC ${SDL2_INCLUDE_DIR})
target_link_libraries(ImGuiSDL2Backend ${SDL2_LIBRARY})

add_library(PlatformInterfaceSDL2
    SdlApplication.cpp
    OpenGLApplication.cpp
    SDLOpenGLBackend.cpp
)

target_include_directories(PlatformInterfaceSDL2 PUBLIC
    ${SDL2_INCLUDE_DIR}
)

target_link_libraries(PlatformInterfaceSDL2
    SdlAssetIO
    ImGuiSDL2Backend
    ${SDL2_LIBRARY}
)

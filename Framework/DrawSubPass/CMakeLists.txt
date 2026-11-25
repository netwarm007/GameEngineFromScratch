# GuiSubPass names backends via RhiBackendName (RhiStartup). Full-engine builds
# define RhiStartup in Framework/Manager first; standalone test configurations
# (e.g. Test/RHI/Vulkan) add DrawSubPass without Manager, so define a fallback
# here once to cover every such configuration.
if(NOT TARGET RhiStartup)
    add_library(RhiStartup "${CMAKE_CURRENT_SOURCE_DIR}/../Manager/RhiStartup.cpp")
    target_include_directories(RhiStartup PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../Manager"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Interface")
endif()

add_library(DrawSubPass
    GeometrySubPass.cpp
    GuiSubPass.cpp
    GuiIconLoader.cpp
    DebugOverlaySubPass.cpp
    SkyBoxSubPass.cpp
    ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui.cpp
    ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_draw.cpp
    ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_demo.cpp 
    ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_widgets.cpp
    ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_tables.cpp
)
target_link_libraries(DrawSubPass PRIVATE ImageDecoders)
# GuiSubPass names backends via RhiBackendName (M5.1 selection menu).
target_link_libraries(DrawSubPass PUBLIC RhiStartup)

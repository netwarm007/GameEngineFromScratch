find_library(COCOA_LIBRARY Cocoa required)
find_library(CV_LIBRARY CoreVideo required)

# Platform Interface library
add_library(PlatformInterface
        CocoaApplication.mm 
        CocoaMetalApplication.mm
        AppDelegate.m
        WindowDelegate.m
        MetalView.mm
        ${PROJECT_SOURCE_DIR}/External/src/imgui/examples/imgui_impl_osx.mm
        ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui.cpp
        ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_draw.cpp
        ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_demo.cpp
        ${PROJECT_SOURCE_DIR}/External/src/imgui/imgui_widgets.cpp
    )
target_link_libraries(PlatformInterface
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        dl
    )

find_library(COCOA_LIBRARY Cocoa required)
find_library(CV_LIBRARY CoreVideo required)
find_library(GAMECONTROLLER_LIBRARY GameController required)

# Platform Interface library
add_library(PlatformInterface
        BundleAssetLoader.mm
        CocoaApplication.mm 
        CocoaMetalApplication.mm
        AppDelegate.m
        WindowDelegate.m
        MetalView.mm
        imgui_impl_osx.mm
    )
target_link_libraries(PlatformInterface
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        ${GAMECONTROLLER_LIBRARY}
        dl
    )

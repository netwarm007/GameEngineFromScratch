find_library(COCOA_LIBRARY Cocoa required)
find_library(CV_LIBRARY CoreVideo required)

# Platform Interface library
add_library(PlatformInterface
        CocoaApplication.mm 
        CocoaMetalApplication.mm
        AppDelegate.m
        WindowDelegate.m
        MetalView.mm
    )
target_link_libraries(PlatformInterface
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        dl
    )

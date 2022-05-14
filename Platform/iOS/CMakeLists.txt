find_library(UIKIT_LIBRARY UIKit required)
find_library(FOUNDATION_LIBRARY Foundation required)

# Platform Interface library
add_library(PlatformInterface
        UIKitApplication.mm 
        AppDelegate.m
        GameViewController.mm
        imgui_impl_ios.mm
    )
target_link_libraries(PlatformInterface
        ${UIKIT_LIBRARY} 
        ${FOUNDATION_LIBRARY}
        dl
    )
find_library(COCOA_LIBRARY Cocoa required)
find_library(CV_LIBRARY CoreVideo required)
find_library(GAMECONTROLLER_LIBRARY GameController required)

# Platform Interface library
set (PlatformSources 
        BundleAssetLoader.mm
        CocoaApplication.mm 
        CocoaMetalApplication.mm
        AppDelegate.m
        WindowDelegate.m
        MetalView.mm
        imgui_impl_osx.mm
)

if(Vulkan_FOUND)
list (APPEND PlatformSources CocoaVulkanApplication.mm)
endif(Vulkan_FOUND)

add_library(PlatformInterface
        ${PlatformSources}
    )
target_link_libraries(PlatformInterface
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        ${GAMECONTROLLER_LIBRARY}
        dl
    )

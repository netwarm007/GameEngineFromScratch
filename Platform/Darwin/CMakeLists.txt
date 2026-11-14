find_library(COCOA_LIBRARY Cocoa required)
find_library(CV_LIBRARY CoreVideo required)
find_library(GAMECONTROLLER_LIBRARY GameController required)

add_platform_asset_io(PlatformAssetIO AssetLoader.mm)

# Platform Interface library
set (PlatformSources 
        CocoaApplication.mm 
        CocoaMetalApplication.mm
        AppDelegate.m
        WindowDelegate.mm
        MetalView.mm
        imgui_impl_osx.mm
)

if(Vulkan_FOUND)
list (APPEND PlatformSources VulkanApplication.mm)
endif(Vulkan_FOUND)

add_library(PlatformInterface
        ${PlatformSources}
    )
target_link_libraries(PlatformInterface
        PlatformAssetIO
        ${COCOA_LIBRARY} 
        ${OPENGL_LIBRARY}
        ${GAMECONTROLLER_LIBRARY}
        dl
    )

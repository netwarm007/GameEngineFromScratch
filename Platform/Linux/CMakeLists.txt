add_platform_asset_io(PlatformAssetIO AssetLoader.cpp)

find_library(XCB_LIBRARY xcb)
find_library(X11_LIBRARY X11)
find_library(X11_XCB_LIBRARY X11-xcb)

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES FreeBSD)
        set (DL_LIBRARY "")
else(${CMAKE_HOST_SYSTEM_NAME} MATCHES FreeBSD)
        find_library(DL_LIBRARY dl)
endif(${CMAKE_HOST_SYSTEM_NAME} MATCHES FreeBSD)

set (PlatformSources 
        XcbApplication.cpp 
        OpenGLApplication.cpp
        ${PROJECT_SOURCE_DIR}/External/src/glad/src/glad_glx.c)

if(Vulkan_FOUND)
list (APPEND PlatformSources VulkanApplication.cpp)
endif(Vulkan_FOUND)

# XCB host pieces for runtime RHI selection builds that keep the SDL OpenGL
# application: excludes the XCB OpenGLApplication and PlatformAssetIO so they
# do not clash with the SDL variants in the same executable.
if(Vulkan_FOUND)
    add_library(PlatformVulkanXcb XcbApplication.cpp VulkanApplication.cpp LinuxHost.cpp)
    target_include_directories(PlatformVulkanXcb PRIVATE ${SDL2_INCLUDE_DIR})
    target_link_libraries(PlatformVulkanXcb
        ${XCB_LIBRARY}
        ${X11_LIBRARY}
        ${X11_XCB_LIBRARY}
        ${DL_LIBRARY}
        ${SDL2_LIBRARY}
        $<$<TARGET_EXISTS:ImGuiSDL2Backend>:ImGuiSDL2Backend>
        $<$<TARGET_EXISTS:VulkanLoader>:VulkanLoader>
    )
endif()

add_library(PlatformInterface
    ${PlatformSources}
)

target_link_libraries(PlatformInterface
        PlatformAssetIO
        ${XCB_LIBRARY} 
        ${X11_LIBRARY} 
        ${X11_XCB_LIBRARY} 
        ${DL_LIBRARY}
        $<$<TARGET_EXISTS:VulkanLoader>:VulkanLoader>
)

target_include_directories(PlatformInterface PRIVATE "${PROJECT_SOURCE_DIR}/External/src/glad/include")

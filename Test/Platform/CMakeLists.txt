set(PLATFORM_TEST_CASES BasicWindowTest)

foreach(TEST_CASE IN LISTS PLATFORM_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
endforeach(TEST_CASE)

if(SDL2_FOUND)
    add_executable(Sdl2WindowTest Sdl2WindowTest.cpp)
    target_link_libraries(Sdl2WindowTest Framework PlatformInterfaceSDL2)
endif(SDL2_FOUND)

set(GRAPHICS_TEST_CASES GraphicsWindowTest)

if(OS_WINDOWS OR OS_MACOS)
    list(APPEND GRAPHICS_TEST_CASES ImGuiTest)
endif(OS_WINDOWS OR OS_MACOS)

if(Vulkan_FOUND)
    set(VULKAN_RHI VulkanRHI)
endif(Vulkan_FOUND)

foreach(TEST_CASE IN LISTS GRAPHICS_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)

    if(OS_MACOS)
        target_link_libraries(${TEST_CASE} Framework PlatformInterface MetalRHI ${VULKAN_RHI})
    elseif(OS_WINDOWS)
        target_link_libraries(${TEST_CASE} Framework PlatformInterface D3dRHI OpenGLRHI ${VULKAN_RHI})
    else()
        target_link_libraries(${TEST_CASE} Framework PlatformInterface OpenGLRHI ${VULKAN_RHI})
    endif(OS_MACOS)
endforeach(TEST_CASE)

if(SDL2_FOUND)
    add_executable(Sdl2OpenGLWindowTest Sdl2OpenGLWindowTest.cpp)
    target_link_libraries(Sdl2OpenGLWindowTest Framework PlatformInterfaceSDL2 OpenGLRHI)

    if(OS_LINUX)
        add_executable(ImGuiTestSDL2 ImGuiTest.cpp)
        target_link_libraries(ImGuiTestSDL2 Framework PlatformInterfaceSDL2 OpenGLRHI ${VULKAN_RHI})
    endif(OS_LINUX)
endif(SDL2_FOUND)

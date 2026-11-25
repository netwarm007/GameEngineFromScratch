include_guard(GLOBAL)

function(myge_add_vulkan_loader root)
    if(TARGET VulkanLoader)
        return()
    endif()
    add_library(VulkanLoader STATIC "${root}/RHI/Vulkan/VulkanLoader.cpp")
    target_include_directories(VulkanLoader PUBLIC
        "${root}/RHI/Vulkan" "${Vulkan_INCLUDE_DIR}")
    target_compile_definitions(VulkanLoader PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 PRIVATE VK_NO_PROTOTYPES)
    target_link_libraries(VulkanLoader PUBLIC ${CMAKE_DL_LIBS})
endfunction()

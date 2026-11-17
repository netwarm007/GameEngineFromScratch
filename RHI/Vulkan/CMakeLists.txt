include("${PROJECT_SOURCE_DIR}/cmake/ImGuiVulkanBackend.cmake")
myge_prepare_imgui_vulkan_backend(imgui_vulkan_backend "${PROJECT_SOURCE_DIR}")

add_library(VulkanRHI
    VulkanRHI.cpp
    VulkanRHIResources.cpp
    VulkanPipelineStateManager.cpp
    VulkanGraphicsManager.cpp
    VulkanGraphicsRendering.cpp
    VulkanGraphicsShadow.cpp
    VulkanGuiRenderer.cpp
    VulkanGraphicsGui.cpp
    ${imgui_vulkan_backend}
)

target_link_libraries(VulkanRHI ${Vulkan_LIBRARIES})
if(OS_MACOS)
target_link_libraries(VulkanRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
endif(OS_MACOS)

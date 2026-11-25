include("${PROJECT_SOURCE_DIR}/cmake/ImGuiVulkanBackend.cmake")
myge_prepare_imgui_vulkan_backend(imgui_vulkan_backend "${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmake/VulkanLoader.cmake")
myge_add_vulkan_loader("${PROJECT_SOURCE_DIR}")

add_library(VulkanRHI
    VulkanRHI.cpp
    VulkanBackend.cpp
    VulkanRHIResources.cpp
    VulkanPipelineStateManager.cpp
    VulkanGraphicsManager.cpp
    VulkanGraphicsRendering.cpp
    VulkanGraphicsShadow.cpp
    VulkanGuiRenderer.cpp
    VulkanGraphicsGui.cpp
    ${imgui_vulkan_backend}
)

target_link_libraries(VulkanRHI VulkanLoader Manager)
target_compile_definitions(VulkanRHI PRIVATE VK_NO_PROTOTYPES)
if(OS_MACOS)
target_link_libraries(VulkanRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
endif(OS_MACOS)

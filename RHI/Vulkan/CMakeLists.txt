add_library(VulkanRHI
    VulkanRHI.cpp
    VulkanRHIResources.cpp
    VulkanPipelineStateManager.cpp
    VulkanGraphicsManager.cpp
    VulkanGraphicsRendering.cpp
    VulkanGraphicsShadow.cpp
    VulkanGuiRenderer.cpp
    VulkanGraphicsGui.cpp
    ${PROJECT_SOURCE_DIR}/External/src/imgui/backends/imgui_impl_vulkan.cpp
)

target_link_libraries(VulkanRHI ${Vulkan_LIBRARIES})
set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/External/src/imgui/backends/imgui_impl_vulkan.cpp
    PROPERTIES COMPILE_DEFINITIONS IMGUI_IMPL_VULKAN_NO_PROTOTYPES)

if(OS_MACOS)
target_link_libraries(VulkanRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
endif(OS_MACOS)

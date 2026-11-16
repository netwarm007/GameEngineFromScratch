add_library(VulkanRHI
    VulkanRHI.cpp
    VulkanPipelineStateManager.cpp
    VulkanGraphicsManager.cpp
    ${PROJECT_SOURCE_DIR}/External/src/imgui/backends/imgui_impl_vulkan.cpp
)

target_include_directories(VulkanRHI PRIVATE
    ${PROJECT_SOURCE_DIR}/External/src/imgui
)
target_link_libraries(VulkanRHI ${Vulkan_LIBRARIES})

if(OS_MACOS)
target_link_libraries(VulkanRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
endif(OS_MACOS)

add_library(VulkanRHI
    VulkanRHI.cpp
    VulkanPipelineStateManager.cpp
    VulkanGraphicsManager.cpp
)

target_link_libraries(VulkanRHI ${Vulkan_LIBRARIES})

if(OS_MACOS)
target_link_libraries(VulkanRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
endif(OS_MACOS)

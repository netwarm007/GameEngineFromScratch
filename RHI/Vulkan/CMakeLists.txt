add_library(VulkanRHI
    VulkanRHI.cpp
)

target_link_libraries(VulkanRHI GeomMath ${Vulkan_LIBRARIES})

if(OS_MACOS)
target_link_libraries(VulkanRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )
endif(OS_MACOS)

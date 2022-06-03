add_library(VulkanRHI
    VulkanRHI.cpp
)

target_link_libraries(VulkanRHI GeomMath ${Vulkan_LIBRARIES})
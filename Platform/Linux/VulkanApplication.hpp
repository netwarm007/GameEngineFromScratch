#pragma once
#include "Vulkan/VulkanRHI.hpp"
#include "XcbApplication.hpp"

namespace My {
class VulkanApplication : public XcbApplication {
public:
    using XcbApplication::XcbApplication;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

private:
    int InitializeGraphicsResources() final;
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
} // namespace My


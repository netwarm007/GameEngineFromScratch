#pragma once
#include "Vulkan/VulkanRHI.hpp"
#include "XcbApplication.hpp"

namespace My {
class VulkanApplication : public XcbApplication {
public:
    using XcbApplication::XcbApplication;

    void CreateMainWindow() final;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

private:
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
} // namespace My


#pragma once
#include "Vulkan/VulkanRHI.hpp"
#include "XcbApplication.hpp"

namespace My {
class VulkanApplication : public XcbApplication {
public:
    using XcbApplication::XcbApplication;

    void Finalize() final;

    void CreateMainWindow() final;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

private:
    VulkanRHI m_Rhi;
};
} // namespace My


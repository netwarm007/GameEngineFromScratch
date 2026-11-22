#pragma once
#include "IRenderBackend.hpp"
#include "Vulkan/VulkanRHI.hpp"
#include "XcbApplication.hpp"

namespace My {
class VulkanApplication : public XcbApplication, public IRenderBackend {
public:
    using XcbApplication::XcbApplication;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

private:
    int InitializeGraphicsResources() final;
    void WaitForGraphicsIdle() final;
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
} // namespace My


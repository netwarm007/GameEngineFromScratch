#pragma once
#include "Vulkan/VulkanRHI.hpp"
#include "WindowsApplication.hpp"

namespace My {
class VulkanApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

   private:
    int InitializeGraphicsResources() final;
    void WaitForGraphicsIdle() final;
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
}  // namespace My

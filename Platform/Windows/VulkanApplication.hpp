#pragma once
#include "Vulkan/VulkanRHI.hpp"
#include "WindowsApplication.hpp"

namespace My {
class VulkanApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    void CreateMainWindow() final;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

   private:
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
}  // namespace My

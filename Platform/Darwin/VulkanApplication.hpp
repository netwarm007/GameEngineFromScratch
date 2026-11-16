#pragma once
#include "CocoaApplication.h"
#include "Vulkan/VulkanRHI.hpp"

namespace My {
class VulkanApplication : public CocoaApplication {
   public:
    using CocoaApplication::CocoaApplication;
    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

   private:
    int InitializeGraphicsResources() final;
    void WaitForGraphicsIdle() final;
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
}  // namespace My

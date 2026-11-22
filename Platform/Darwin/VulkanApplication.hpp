#pragma once
#include "CocoaApplication.h"
#include "IRenderBackend.hpp"
#include "Vulkan/VulkanRHI.hpp"

namespace My {
class VulkanApplication : public CocoaApplication, public IRenderBackend {
   public:
    using CocoaApplication::CocoaApplication;
    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

   private:
    int InitializeGraphicsResources() final;
    int InitializeGuiPlatform() final;
    void WaitForGraphicsIdle() final;
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
}  // namespace My

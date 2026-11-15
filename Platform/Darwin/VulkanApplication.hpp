#pragma once
#include "CocoaMetalApplication.h"
#include "Vulkan/VulkanRHI.hpp"

namespace My {
class VulkanApplication : public CocoaMetalApplication {
   public:
    using CocoaMetalApplication::CocoaMetalApplication;
    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    void CreateMainWindow() final;

    VulkanRHI& GetRHI() { return m_Rhi; }

   private:
    void FinalizeGraphicsResources() final;
    VulkanRHI m_Rhi;
};
}  // namespace My

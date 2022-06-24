#pragma once
#include "CocoaMetalApplication.h"
#include "Vulkan/VulkanRHI.hpp"

namespace My {
class CocoaVulkanApplication : public CocoaMetalApplication {
   public:
    using CocoaMetalApplication::CocoaMetalApplication;
    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    void CreateMainWindow() final;

    void Finalize() final;

   private:
    VulkanRHI m_Rhi;
};
}  // namespace My

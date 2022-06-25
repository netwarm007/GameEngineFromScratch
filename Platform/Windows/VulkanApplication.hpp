#include "Vulkan/VulkanRHI.hpp"
#include "WindowsApplication.hpp"

namespace My {
class VulkanApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    void Finalize() final;

    void CreateMainWindow() final;

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Rhi; }

   private:
    VulkanRHI m_Rhi;
};
}  // namespace My

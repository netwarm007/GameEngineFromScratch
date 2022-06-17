#include "WindowsApplication.hpp"
#include "vulkan/vulkan.hpp"

namespace My {
class VulkanApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;
    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);
};
}  // namespace My

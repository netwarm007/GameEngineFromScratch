#pragma once

#include <vulkan/vulkan_core.h>

namespace My {

class VulkanLoader {
   public:
    explicit VulkanLoader(const char* libraryPath = nullptr);
    ~VulkanLoader();
    VulkanLoader(const VulkanLoader&) = delete;
    VulkanLoader& operator=(const VulkanLoader&) = delete;

    PFN_vkGetInstanceProcAddr GetInstanceProcAddr() const {
        return m_getInstanceProcAddr;
    }
    PFN_vkVoidFunction GetInstanceFunction(VkInstance instance,
                                           const char* name) const;
    PFN_vkVoidFunction GetExportedFunction(const char* name) const;

   private:
    void Close();
    void* m_module = nullptr;
    PFN_vkGetInstanceProcAddr m_getInstanceProcAddr = nullptr;
};

// Load only on demand; retain the loader until process exit, not just device exit.
VulkanLoader& GetVulkanLoader();

}  // namespace My

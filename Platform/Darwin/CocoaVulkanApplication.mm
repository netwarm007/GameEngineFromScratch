#import <Cocoa/Cocoa.h>
#include "CocoaVulkanApplication.h"

using namespace My;

using VkMetalSurfaceCreateFlagsEXT = VkFlags;

typedef struct VkMetalSurfaceCreateInfoEXT
{
    VkStructureType                 sType;
    const void*                     pNext;
    VkMetalSurfaceCreateFlagsEXT    flags;
    const void*                     pLayer;
} VkMetalSurfaceCreateInfoEXT;

typedef VkResult (VKAPI_PTR *PFN_vkCreateMetalSurfaceEXT)(VkInstance, const VkMetalSurfaceCreateInfoEXT*, const VkAllocationCallbacks*, VkSurfaceKHR*);

VkResult CocoaVulkanApplication::CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface) {
    @autoreleasepool {
        VkResult err;

        VkMetalSurfaceCreateInfoEXT sci;

        PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT;
        vkCreateMetalSurfaceEXT = (PFN_vkCreateMetalSurfaceEXT)
            vkGetInstanceProcAddr(instance, "vkCreateMetalSurfaceEXT");
        if (!vkCreateMetalSurfaceEXT) {
            printf("Cocoa: Vulkan instance missing VK_EXT_metal_surface extension\n");
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }

        memset(&sci, 0, sizeof(sci));
        sci.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        sci.pLayer = [[m_pWindow contentView] layer];

        err = vkCreateMetalSurfaceEXT(instance, &sci, nullptr, &surface);

        return err;
    }
}

void CocoaVulkanApplication::CreateMainWindow() {
    CocoaMetalApplication::CreateMainWindow();

    // 设置回调函数
    auto getFramebufferSize = [this](int& width, int& height) {
        GetFramebufferSize(width, height);
    };

    // 设置回调函数
    m_Rhi.setFramebufferSizeQueryCB(getFramebufferSize);

    // 创建实例
    m_Rhi.createInstance(extensions);

    // 开启调试层（对发行版不起作用）
    m_Rhi.setupDebugMessenger();

    // 创建（连接）画布
    {
        auto createWindowSurface = [this](const vk::Instance& instance,
                                          vk::SurfaceKHR& surface) {
            VkSurfaceKHR _surface;
            assert(instance);

            if (CreateWindowSurface(instance, _surface) != VK_SUCCESS) {
                throw std::runtime_error("faild to create window surface!");
            }

            surface = _surface;
        };

        m_Rhi.createSurface(createWindowSurface);
    }

    // 枚举物理设备并选择
    m_Rhi.pickPhysicalDevice();

    // 创建逻辑设备
    m_Rhi.createLogicalDevice();

    // 创建 SwapChain
    m_Rhi.createSwapChain();
}

void CocoaVulkanApplication::Finalize() {
    m_Rhi.destroyAll();

    CocoaVulkanApplication::Finalize();
}
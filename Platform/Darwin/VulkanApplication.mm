#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include "VulkanApplication.hpp"
#include <vulkan/vulkan_metal.h>

using namespace My;

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface) {
    @autoreleasepool {
        VkResult err;

        VkMetalSurfaceCreateInfoEXT sci{};

        PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT;
        vkCreateMetalSurfaceEXT = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateMetalSurfaceEXT"));
        if (!vkCreateMetalSurfaceEXT) {
            printf("Cocoa: Vulkan instance missing VK_EXT_metal_surface extension\n");
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }

        sci.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        sci.pLayer = static_cast<CAMetalLayer*>([[m_pWindow contentView] layer]);

        err = vkCreateMetalSurfaceEXT(instance, &sci, nullptr, &surface);

        return err;
    }
}

int VulkanApplication::InitializeGraphicsResources() {
    if (!GetMainWindowHandler()) {
        fprintf(stderr, "Vulkan initialization requires a native window\n");
        return -1;
    }
    @autoreleasepool {
        NSView* view = [[NSView alloc] initWithFrame:[[m_pWindow contentView] bounds]];
        view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        view.wantsLayer = YES;
        view.layer = [CAMetalLayer layer];
        [m_pWindow setContentView:view];
        [view release];
    }

    // 设置回调函数
    auto getFramebufferSize = [this](uint32_t& width, uint32_t& height) {
        if ([m_pWindow isMiniaturized]) {
            width = height = 0;
            return;
        }
        GetFramebufferSize(width, height);
        CAMetalLayer* layer = static_cast<CAMetalLayer*>([[m_pWindow contentView] layer]);
        layer.contentsScale = m_pWindow.backingScaleFactor;
        layer.drawableSize = CGSizeMake(width, height);
    };

    // 设置回调函数
    m_Rhi.setFramebufferSizeQueryCB(getFramebufferSize);

    // 创建实例
    std::vector<const char*> extensions = {
        "VK_KHR_surface", "VK_EXT_metal_surface",
        "VK_KHR_portability_enumeration",
        "VK_KHR_get_physical_device_properties2"};

    m_Rhi.SetGetGfxConfigCB([this] { return GetConfiguration(); });
    m_Rhi.Initialize(extensions, [this](const vk::Instance& instance, vk::SurfaceKHR& surface) {
        VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
        const auto result = CreateWindowSurface(instance, nativeSurface);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Cocoa Vulkan surface creation failed: " +
                                     std::to_string(result));
        }
        surface = nativeSurface;
    });
    return 0;
}

void VulkanApplication::WaitForGraphicsIdle() { m_Rhi.WaitGPU(); }

void VulkanApplication::FinalizeGraphicsResources() {
    m_Rhi.destroyAll();
}
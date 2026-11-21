#include "VulkanApplication.hpp"
#include <cstdio>
#include <exception>
#include <vulkan/vulkan_xcb.h>

using namespace My;

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance,
                                                VkSurfaceKHR& surface) {
    VkXcbSurfaceCreateInfoKHR sci{};
    PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR;

    vkCreateXcbSurfaceKHR = reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(
        GetVulkanLoader().GetInstanceFunction(instance, "vkCreateXcbSurfaceKHR"));

    if (!vkCreateXcbSurfaceKHR) {
        fprintf(stderr,
                "Xcb: Vulkan instance missing VK_KHR_xcb_surface extension");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    sci.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    sci.connection = m_pConn;
    sci.window = m_XWindow;
    return vkCreateXcbSurfaceKHR(instance, &sci, nullptr, &surface);
}

int VulkanApplication::InitializeGraphicsResources() {
    if (!m_pConn || m_XWindow == XCB_WINDOW_NONE) {
        fprintf(stderr, "Vulkan initialization requires a native window\n");
        return -1;
    }

    std::vector<const char*> extensions = {"VK_KHR_surface",
                                           "VK_KHR_xcb_surface"};

    // 设置回调函数
    auto getFramebufferSize = [this](uint32_t& width, uint32_t& height) {
        GetFramebufferSize(width, height);
    };

    // 设置回调函数
    m_Rhi.setFramebufferSizeQueryCB(getFramebufferSize);

    m_Rhi.SetGetGfxConfigCB([this] { return GetConfiguration(); });
    try {
        m_Rhi.Initialize(extensions, [this](const vk::Instance& instance, vk::SurfaceKHR& surface) {
            VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
            const auto result = CreateWindowSurface(instance, nativeSurface);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Xcb Vulkan surface creation failed: " +
                                         std::to_string(result));
            }
            surface = nativeSurface;
        });
    } catch (const std::exception& error) {
        // The RHI has already rolled back its partial state.
        fprintf(stderr, "Vulkan initialization failed: %s\n", error.what());
        return -1;
    }
    return 0;
}

void VulkanApplication::WaitForGraphicsIdle() { m_Rhi.WaitGPU(); }

void VulkanApplication::FinalizeGraphicsResources() {
    // 销毁 Vulkan 上下文
    m_Rhi.destroyAll();
}

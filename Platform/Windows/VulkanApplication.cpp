#include "VulkanApplication.hpp"
#include <vulkan/vulkan_win32.h>

using namespace My;

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance,
                                                VkSurfaceKHR& surface) {
    VkWin32SurfaceCreateInfoKHR sci{};
    PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;

    vkCreateWin32SurfaceKHR =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(vkGetInstanceProcAddr(
            instance, "vkCreateWin32SurfaceKHR"));
    if (!vkCreateWin32SurfaceKHR) {
        fprintf(
            stderr,
            "Win32: Vulkan instance missing VK_KHR_win32_surface extension");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = m_hInstance;
    sci.hwnd = m_hWnd;

    return vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface);
}

int VulkanApplication::InitializeGraphicsResources() {
    if (!GetMainWindowHandler()) {
        fprintf(stderr, "Vulkan initialization requires a native window\n");
        return -1;
    }

    std::vector<const char*> extensions = {"VK_KHR_surface",
                                           "VK_KHR_win32_surface"};

    // 设置回调函数
    auto getFramebufferSize = [this](uint32_t& width, uint32_t& height) {
        GetFramebufferSize(width, height);
    };

    m_Rhi.setFramebufferSizeQueryCB(getFramebufferSize);
    m_Rhi.SetGetGfxConfigCB([this] { return GetConfiguration(); });
    m_Rhi.Initialize(extensions, [this](const vk::Instance& instance, vk::SurfaceKHR& surface) {
        VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
        const auto result = CreateWindowSurface(instance, nativeSurface);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Win32 Vulkan surface creation failed: " +
                                     std::to_string(result));
        }
        surface = nativeSurface;
    });
    return 0;
}

void VulkanApplication::WaitForGraphicsIdle() { m_Rhi.WaitGPU(); }

void VulkanApplication::FinalizeGraphicsResources() {
    // 销毁Vulkan上下文
    m_Rhi.destroyAll();
}

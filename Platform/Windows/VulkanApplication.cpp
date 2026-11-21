#include "VulkanApplication.hpp"
#include <cstdio>
#include <exception>
#include <vulkan/vulkan_win32.h>
#include "imgui.h"

using namespace My;

int VulkanApplication::InitializeGuiPlatform() {
    const auto result = WindowsApplication::InitializeGuiPlatform();
    if (result != 0 || !ImGui::GetCurrentContext()) return result;
    ImGui::GetPlatformIO().Platform_CreateVkSurface =
        [](ImGuiViewport* viewport, ImU64 instance, const void* allocator, ImU64* output) -> int {
        const auto nativeInstance = reinterpret_cast<VkInstance>(instance);
        const auto create = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            GetVulkanLoader().GetInstanceFunction(nativeInstance, "vkCreateWin32SurfaceKHR"));
        if (!create) return VK_ERROR_EXTENSION_NOT_PRESENT;
        const auto window = static_cast<HWND>(viewport->PlatformHandleRaw);
        VkWin32SurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        info.hwnd = window;
        info.hinstance = reinterpret_cast<HINSTANCE>(GetWindowLongPtr(window, GWLP_HINSTANCE));
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        const auto status = create(nativeInstance, &info,
            static_cast<const VkAllocationCallbacks*>(allocator), &surface);
        if (status == VK_SUCCESS) *output = reinterpret_cast<ImU64>(surface);
        return status;
    };
    return 0;
}

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance,
                                                VkSurfaceKHR& surface) {
    VkWin32SurfaceCreateInfoKHR sci{};
    PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;

    vkCreateWin32SurfaceKHR =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(GetVulkanLoader().GetInstanceFunction(
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
    try {
        m_Rhi.Initialize(extensions, [this](const vk::Instance& instance, vk::SurfaceKHR& surface) {
            VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
            const auto result = CreateWindowSurface(instance, nativeSurface);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Win32 Vulkan surface creation failed: " +
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
    // 销毁Vulkan上下文
    m_Rhi.destroyAll();
}

#include "VulkanApplication.hpp"

#include <cstdio>
#include <vulkan/vulkan_win32.h>

#include "imgui.h"

using namespace My;

namespace My {
int InstallWin32VkSurfaceCallback() {
    if (!ImGui::GetCurrentContext()) return 0;
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

std::unique_ptr<VulkanBackend> MakeWin32VulkanBackend(WindowsApplication& host) {
    return std::make_unique<VulkanBackend>(
        host, host,
        std::vector<const char*>{"VK_KHR_surface", "VK_KHR_win32_surface"},
        [&host](vk::Instance instance, VkSurfaceKHR& surface) -> VkResult {
            PFN_vkCreateWin32SurfaceKHR create =
                reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
                    GetVulkanLoader().GetInstanceFunction(instance, "vkCreateWin32SurfaceKHR"));
            if (!create) {
                fprintf(stderr, "Win32: Vulkan instance missing VK_KHR_win32_surface extension\n");
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            HWND window = (HWND)host.GetMainWindowHandler();
            VkWin32SurfaceCreateInfoKHR sci{};
            sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
            sci.hinstance = reinterpret_cast<HINSTANCE>(GetWindowLongPtr(window, GWLP_HINSTANCE));
            sci.hwnd = window;
            return create(instance, &sci, nullptr, &surface);
        },
        [] { return InstallWin32VkSurfaceCallback(); });
}
}  // namespace My

VulkanApplication::VulkanApplication()
    : WindowsApplication(), m_Backend(MakeWin32VulkanBackend(*this)) {}

VulkanApplication::VulkanApplication(GfxConfiguration& cfg)
    : WindowsApplication(cfg), m_Backend(MakeWin32VulkanBackend(*this)) {}

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

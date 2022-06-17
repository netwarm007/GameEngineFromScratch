#include "VulkanApplication.hpp"

using namespace My;

using VkWin32SurfaceCreateFlagsKHR = VkFlags;

typedef struct VkWin32SurfaceCreateInfoKHR {
    VkStructureType sType;
    const void* pNext;
    VkWin32SurfaceCreateFlagsKHR flags;
    HINSTANCE hinstance;
    HWND hwnd;
} VkWin32SurfaceCreateInfoKHR;

typedef VkResult(APIENTRY* PFN_vkCreateWin32SurfaceKHR)(
    VkInstance, const VkWin32SurfaceCreateInfoKHR*,
    const VkAllocationCallbacks*, VkSurfaceKHR*);

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance,
                                                VkSurfaceKHR& surface) {
    VkResult err;
    VkWin32SurfaceCreateInfoKHR sci;
    PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;

    vkCreateWin32SurfaceKHR =
        (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(
            instance, "vkCreateWin32SurfaceKHR");
    if (!vkCreateWin32SurfaceKHR) {
        fprintf(
            stderr,
            "Win32: Vulkan instance missing VK_KHR_win32_surface extension");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = m_hInstance;
    sci.hwnd = m_hWnd;

    err = vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface);
    assert(!err && "Win32: Failed to create Vulkan surface.");

    return err;
}

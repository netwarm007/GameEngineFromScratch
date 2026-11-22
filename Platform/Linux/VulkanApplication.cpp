#include "VulkanApplication.hpp"

#include <cstdio>
#include <vulkan/vulkan_xcb.h>

using namespace My;

VulkanApplication::VulkanApplication()
    : XcbApplication(),
      m_Backend(*this, *this,
                {"VK_KHR_surface", "VK_KHR_xcb_surface"},
                [this](vk::Instance instance, VkSurfaceKHR& surface) {
                    return CreateWindowSurface(instance, surface);
                }) {}

VulkanApplication::VulkanApplication(GfxConfiguration& cfg)
    : XcbApplication(cfg),
      m_Backend(*this, *this,
                {"VK_KHR_surface", "VK_KHR_xcb_surface"},
                [this](vk::Instance instance, VkSurfaceKHR& surface) {
                    return CreateWindowSurface(instance, surface);
                }) {}

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

std::unique_ptr<VulkanBackend> My::MakeXcbVulkanBackend(
    IApplication& application, IPlatformHost& platform,
    xcb_connection_t* connection) {
    return std::make_unique<VulkanBackend>(
        application, platform,
        std::vector<const char*>{"VK_KHR_surface", "VK_KHR_xcb_surface"},
        [&application, connection](vk::Instance instance, VkSurfaceKHR& surface) -> VkResult {
            PFN_vkCreateXcbSurfaceKHR create =
                reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(
                    GetVulkanLoader().GetInstanceFunction(instance, "vkCreateXcbSurfaceKHR"));
            if (!create) {
                fprintf(stderr, "Xcb: Vulkan instance missing VK_KHR_xcb_surface extension\n");
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            VkXcbSurfaceCreateInfoKHR sci{};
            sci.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
            sci.connection = connection;
            sci.window = static_cast<xcb_window_t>(
                reinterpret_cast<uintptr_t>(application.GetMainWindowHandler()));
            return create(instance, &sci, nullptr, &surface);
        });
}

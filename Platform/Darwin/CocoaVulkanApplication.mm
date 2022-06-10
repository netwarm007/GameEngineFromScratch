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

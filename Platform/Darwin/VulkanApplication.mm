#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include "VulkanApplication.hpp"
#include <cstdio>
#include <vulkan/vulkan_metal.h>
#include "imgui.h"

using namespace My;

namespace {
ImVec2 (*cocoaGetWindowFramebufferScale)(ImGuiViewport*) = nullptr;

ImVec2 GetVulkanWindowFramebufferScale(ImGuiViewport* viewport) {
    const auto scale = cocoaGetWindowFramebufferScale(viewport);
    @autoreleasepool {
        NSWindow* window = static_cast<NSWindow*>(viewport->PlatformHandleRaw);
        NSView* view = window.contentView;
        if ([view.layer isKindOfClass:[CAMetalLayer class]]) {
            CAMetalLayer* layer = static_cast<CAMetalLayer*>(view.layer);
            layer.contentsScale = window.backingScaleFactor;
            const auto size = [view convertRectToBacking:view.bounds].size;
            layer.drawableSize = CGSizeMake(size.width, size.height);
        }
    }
    return scale;
}

int InstallCocoaVkGuiOverrides() {
    if (!ImGui::GetCurrentContext()) return 0;
    auto& platform = ImGui::GetPlatformIO();
    platform.Platform_CreateVkSurface =
        [](ImGuiViewport* viewport, ImU64 instance, const void* allocator, ImU64* output) -> int {
        @autoreleasepool {
            const auto nativeInstance = reinterpret_cast<VkInstance>(instance);
            const auto create = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
                GetVulkanLoader().GetInstanceFunction(nativeInstance, "vkCreateMetalSurfaceEXT"));
            if (!create) return VK_ERROR_EXTENSION_NOT_PRESENT;
            NSWindow* window = static_cast<NSWindow*>(viewport->PlatformHandleRaw);
            NSView* view = window.contentView;
            view.wantsLayer = YES;
            auto* layer = [CAMetalLayer layer];
            layer.contentsScale = window.backingScaleFactor;
            const auto size = [view convertRectToBacking:view.bounds].size;
            layer.drawableSize = CGSizeMake(size.width, size.height);
            view.layer = layer;
            VkMetalSurfaceCreateInfoEXT info{};
            info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
            info.pLayer = layer;
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            const auto status = create(nativeInstance, &info,
                static_cast<const VkAllocationCallbacks*>(allocator), &surface);
            if (status == VK_SUCCESS) *output = reinterpret_cast<ImU64>(surface);
            return status;
        }
    };
    cocoaGetWindowFramebufferScale = platform.Platform_GetWindowFramebufferScale;
    if (cocoaGetWindowFramebufferScale) {
        platform.Platform_GetWindowFramebufferScale = GetVulkanWindowFramebufferScale;
    }
    return 0;
}
}  // namespace

std::unique_ptr<VulkanBackend> My::MakeCocoaVulkanBackend(CocoaApplication& host) {
    return std::make_unique<VulkanBackend>(
        host, host,
        std::vector<const char*>{"VK_KHR_surface", "VK_EXT_metal_surface",
                                 "VK_KHR_portability_enumeration",
                                 "VK_KHR_get_physical_device_properties2"},
        [&host](vk::Instance instance, VkSurfaceKHR& surface) -> VkResult {
            @autoreleasepool {
                PFN_vkCreateMetalSurfaceEXT create =
                    reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
                        GetVulkanLoader().GetInstanceFunction(instance, "vkCreateMetalSurfaceEXT"));
                if (!create) {
                    printf("Cocoa: Vulkan instance missing VK_EXT_metal_surface extension\n");
                    return VK_ERROR_EXTENSION_NOT_PRESENT;
                }
                NSWindow* window = (NSWindow*)host.GetMainWindowHandler();
                VkMetalSurfaceCreateInfoEXT sci{};
                sci.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
                sci.pLayer = static_cast<CAMetalLayer*>([[window contentView] layer]);
                return create(instance, &sci, nullptr, &surface);
            }
        },
        [] { return InstallCocoaVkGuiOverrides(); },
        [&host](uint32_t& width, uint32_t& height) {
            NSWindow* window = (NSWindow*)host.GetMainWindowHandler();
            if ([window isMiniaturized]) {
                width = height = 0;
                return;
            }
            host.GetFramebufferSize(width, height);
            CAMetalLayer* layer =
                static_cast<CAMetalLayer*>([[window contentView] layer]);
            layer.contentsScale = window.backingScaleFactor;
            layer.drawableSize = CGSizeMake(width, height);
        },
        [&host]() -> int {
            @autoreleasepool {
                NSWindow* window = (NSWindow*)host.GetMainWindowHandler();
                NSView* view = [[NSView alloc]
                    initWithFrame:[[window contentView] bounds]];
                view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
                view.wantsLayer = YES;
                view.layer = [CAMetalLayer layer];
                [window setContentView:view];
                [view release];
            }
            return 0;
        });
}

VulkanApplication::VulkanApplication()
    : CocoaApplication(), m_Backend(MakeCocoaVulkanBackend(*this)) {}

VulkanApplication::VulkanApplication(GfxConfiguration& cfg)
    : CocoaApplication(cfg), m_Backend(MakeCocoaVulkanBackend(*this)) {}

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface) {
    @autoreleasepool {
        VkResult err;

        VkMetalSurfaceCreateInfoEXT sci{};

        PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT;
        vkCreateMetalSurfaceEXT = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
            GetVulkanLoader().GetInstanceFunction(instance, "vkCreateMetalSurfaceEXT"));
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

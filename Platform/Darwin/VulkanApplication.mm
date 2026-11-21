#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include "VulkanApplication.hpp"
#include <cstdio>
#include <exception>
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
}  // namespace

int VulkanApplication::InitializeGuiPlatform() {
    const auto result = CocoaApplication::InitializeGuiPlatform();
    if (result != 0 || !ImGui::GetCurrentContext()) return result;
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
    try {
        m_Rhi.Initialize(extensions, [this](const vk::Instance& instance, vk::SurfaceKHR& surface) {
            VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
            const auto result = CreateWindowSurface(instance, nativeSurface);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Cocoa Vulkan surface creation failed: " +
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
    m_Rhi.destroyAll();
}
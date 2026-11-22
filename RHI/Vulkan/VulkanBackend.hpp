#pragma once

#include <functional>
#include <vector>

#include "IRenderBackend.hpp"
#include "VulkanRHI.hpp"

namespace My {
class IApplication;
class IPlatformHost;

// Session-owned Vulkan backend lifecycle against a persistent platform host
// (M4.1 detachment). Platform specifics (native surface creation and the
// ImGui secondary-viewport surface callback) are injected by the pairing
// host/factory, so one class serves Win32, XCB, and Cocoa.
class VulkanBackend : public IRenderBackend {
   public:
    using SurfaceFactory =
        std::function<VkResult(vk::Instance instance, VkSurfaceKHR& surface)>;
    using AfterGuiBind = std::function<int()>;
    using FramebufferSizeQuery = std::function<void(uint32_t&, uint32_t&)>;
    using PrepareDrawable = std::function<int()>;

    VulkanBackend(IApplication& application, IPlatformHost& platform,
                  std::vector<const char*> extensions, SurfaceFactory surface,
                  AfterGuiBind afterGuiBind = {},
                  FramebufferSizeQuery framebufferSizeQuery = {},
                  PrepareDrawable prepareDrawable = {});

    VulkanRHI& GetRHI() { return m_Rhi; }

    void CreateMainWindow() override;
    int InitializeGraphicsResources() override;
    void WaitForGraphicsIdle() override;
    void FinalizeGraphicsResources() override;
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;

   private:
    IApplication& m_Application;
    IPlatformHost& m_Platform;
    std::vector<const char*> m_Extensions;
    SurfaceFactory m_Surface;
    AfterGuiBind m_AfterGuiBind;
    FramebufferSizeQuery m_FramebufferSizeQuery;
    PrepareDrawable m_PrepareDrawable;
    VulkanRHI m_Rhi;
};
}  // namespace My
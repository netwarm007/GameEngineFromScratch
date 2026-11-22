#pragma once
#include <functional>
#include <memory>

#include "IRenderBackend.hpp"
#include "IApplication.hpp"
#include "IPlatformHost.hpp"
#include "Vulkan/VulkanBackend.hpp"
#include "XcbApplication.hpp"

namespace My {
// Legacy compile-time application: the XCB host plus an owned Vulkan
// backend. The runtime path composes the host and backend separately.
class VulkanApplication : public XcbApplication, public IRenderBackend {
   public:
    VulkanApplication();
    explicit VulkanApplication(GfxConfiguration& cfg);

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Backend.GetRHI(); }

    void CreateMainWindow() override { XcbApplication::CreateMainWindow(); }
    int InitializeGuiPlatform() override {
        return m_Backend.InitializeGuiPlatform();
    }
    void FinalizeGuiPlatform() override { m_Backend.FinalizeGuiPlatform(); }

   private:
    int InitializeGraphicsResources() final {
        return m_Backend.InitializeGraphicsResources();
    }
    void WaitForGraphicsIdle() final { m_Backend.WaitForGraphicsIdle(); }
    void FinalizeGraphicsResources() final {
        m_Backend.FinalizeGraphicsResources();
    }

    VulkanBackend m_Backend;
};

// Runtime pairing: an XCB-window host (dual-mode Linux host in XCB mode or
// the plain XCB application) + Vulkan backend. The connection is queried
// lazily at surface creation: the dual-mode host only opens it when its
// XCB window is created, after composition.
std::unique_ptr<VulkanBackend> MakeXcbVulkanBackend(
    IApplication& application, IPlatformHost& platform,
    std::function<xcb_connection_t*()> connection);
}  // namespace My

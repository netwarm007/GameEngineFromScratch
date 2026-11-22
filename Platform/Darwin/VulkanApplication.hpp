#pragma once
#include <memory>

#include "CocoaApplication.h"
#include "IRenderBackend.hpp"
#include "Vulkan/VulkanBackend.hpp"

namespace My {
// Legacy compile-time application: the Cocoa host plus an owned Vulkan
// backend. The runtime path composes the host and backend separately.
class VulkanApplication : public CocoaApplication, public IRenderBackend {
   public:
    VulkanApplication();
    explicit VulkanApplication(GfxConfiguration& cfg);

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Backend->GetRHI(); }

    void CreateMainWindow() override { CocoaApplication::CreateMainWindow(); }
    void FinalizeGuiPlatform() final { m_Backend->FinalizeGuiPlatform(); }

   private:
    int InitializeGraphicsResources() final {
        return m_Backend->InitializeGraphicsResources();
    }
    int InitializeGuiPlatform() final {
        return m_Backend->InitializeGuiPlatform();
    }
    void WaitForGraphicsIdle() final { m_Backend->WaitForGraphicsIdle(); }
    void FinalizeGraphicsResources() final {
        m_Backend->FinalizeGraphicsResources();
    }

    std::unique_ptr<VulkanBackend> m_Backend;
};

// Runtime pairing: Cocoa host + Vulkan backend (shared with the legacy app).
std::unique_ptr<VulkanBackend> MakeCocoaVulkanBackend(CocoaApplication& host);
}  // namespace My

#pragma once
#include <memory>

#include "CocoaApplication.h"
#include "IRenderBackend.hpp"
#include "Vulkan/VulkanBackend.hpp"

namespace My {
// Legacy compile-time application: the Cocoa host plus an owned Vulkan
// backend. The runtime path composes the host and backend separately.
class VulkanApplication : public CocoaApplication, _implements_ IRenderBackend {
   public:
    VulkanApplication();
    explicit VulkanApplication(GfxConfiguration& cfg);

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Backend->GetRHI(); }

    void CreateMainWindow() override { CocoaApplication::CreateMainWindow(); }
    // Qualified base calls: delegating to the backend here would recurse
    // (backend -> IPlatformHost& -> this override -> backend -> ...).
    void FinalizeGuiPlatform() final { CocoaApplication::FinalizeGuiPlatform(); }

   private:
    int InitializeGraphicsResources() final {
        return m_Backend->InitializeGraphicsResources();
    }
    int InitializeGuiPlatform() final {
        const int result = CocoaApplication::InitializeGuiPlatform();
        if (result != 0) return result;
        return m_Backend->BindGuiOverrides();
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

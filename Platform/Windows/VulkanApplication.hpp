#pragma once
#include <memory>

#include "IRenderBackend.hpp"
#include "Vulkan/VulkanBackend.hpp"
#include "WindowsApplication.hpp"

namespace My {
// Legacy compile-time application: the Win32 host plus an owned Vulkan
// backend. The runtime path composes the host and backend separately.
class VulkanApplication : public WindowsApplication, public IRenderBackend {
   public:
    VulkanApplication();
    explicit VulkanApplication(GfxConfiguration& cfg);

    VkResult CreateWindowSurface(vk::Instance instance, VkSurfaceKHR& surface);

    VulkanRHI& GetRHI() { return m_Backend->GetRHI(); }

    void CreateMainWindow() override { WindowsApplication::CreateMainWindow(); }

   private:
    int InitializeGraphicsResources() final {
        return m_Backend->InitializeGraphicsResources();
    }
    void FinalizeGuiPlatform() final { WindowsApplication::FinalizeGuiPlatform(); }
    int InitializeGuiPlatform() final {
        const int result = WindowsApplication::InitializeGuiPlatform();
        if (result != 0) return result;
        return m_Backend->BindGuiOverrides();
    }
    void WaitForGraphicsIdle() final { m_Backend->WaitForGraphicsIdle(); }
    void FinalizeGraphicsResources() final {
        m_Backend->FinalizeGraphicsResources();
    }

    std::unique_ptr<VulkanBackend> m_Backend;
};

// Runtime pairing: Win32 host + Vulkan backend (shared with the legacy app).
std::unique_ptr<VulkanBackend> MakeWin32VulkanBackend(WindowsApplication& host);
}  // namespace My

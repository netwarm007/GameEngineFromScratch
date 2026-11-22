#include "VulkanBackend.hpp"

#include <cstdio>
#include <exception>

#include "IApplication.hpp"
#include "IPlatformHost.hpp"

using namespace My;

VulkanBackend::VulkanBackend(IApplication& application, IPlatformHost& platform,
                             std::vector<const char*> extensions,
                             SurfaceFactory surface, AfterGuiBind afterGuiBind,
                             FramebufferSizeQuery framebufferSizeQuery,
                             PrepareDrawable prepareDrawable)
    : m_Application(application),
      m_Platform(platform),
      m_Extensions(std::move(extensions)),
      m_Surface(std::move(surface)),
      m_AfterGuiBind(std::move(afterGuiBind)),
      m_FramebufferSizeQuery(std::move(framebufferSizeQuery)),
      m_PrepareDrawable(std::move(prepareDrawable)) {}

void VulkanBackend::CreateMainWindow() { m_Application.CreateMainWindow(); }

int VulkanBackend::InitializeGraphicsResources() {
    if (!m_Application.GetMainWindowHandler()) {
        fprintf(stderr, "Vulkan initialization requires a native window\n");
        return -1;
    }
    if (m_PrepareDrawable) {
        const int prepared = m_PrepareDrawable();
        if (prepared != 0) return prepared;
    }

    if (m_FramebufferSizeQuery) {
        m_Rhi.setFramebufferSizeQueryCB(m_FramebufferSizeQuery);
    } else {
        m_Rhi.setFramebufferSizeQueryCB([this](uint32_t& width, uint32_t& height) {
            m_Application.GetFramebufferSize(width, height);
        });
    }
    m_Rhi.SetGetGfxConfigCB([this] { return m_Application.GetConfiguration(); });
    try {
        m_Rhi.Initialize(m_Extensions,
                         [this](const vk::Instance& instance, vk::SurfaceKHR& surface) {
            VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
            const auto result = m_Surface(instance, nativeSurface);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Vulkan surface creation failed: " +
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

void VulkanBackend::WaitForGraphicsIdle() { m_Rhi.WaitGPU(); }

void VulkanBackend::FinalizeGraphicsResources() { m_Rhi.destroyAll(); }

int VulkanBackend::InitializeGuiPlatform() {
    const auto result = m_Platform.InitializeGuiPlatform();
    if (result != 0 || !m_AfterGuiBind) return result;
    return m_AfterGuiBind();
}

void VulkanBackend::FinalizeGuiPlatform() { m_Platform.FinalizeGuiPlatform(); }
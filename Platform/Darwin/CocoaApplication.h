#pragma once
#include "BaseApplication.hpp"
#include "IPlatformHost.hpp"
#include "portable.hpp"

OBJC_CLASS(NSWindow);

namespace My {
class CocoaApplication : public BaseApplication, public IPlatformHost {
   public:
    using BaseApplication::BaseApplication;

    // One cycle of the main loop
    void Tick() override;

    void ProcessEvents() override;

    void* GetMainWindowHandler() override;

    void CreateMainWindow() override;

    void GetFramebufferSize(uint32_t& width, uint32_t& height) override;

    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void FinalizePlatformResources() override;

    void SetActiveBackend(IRenderBackend* backend) override {
        m_pActiveBackend = backend;
    }

    // Headless `-screenshot` runs without `-screenshotWaitFor` are paced to
    // 60 ticks/s so timed captures see vsync-like simulation time (physics
    // advances a fixed step per tick).
    void SetHeadlessPacing(bool enabled) { m_HeadlessPacing = enabled; }

   protected:
    NSWindow* m_pWindow = nullptr;
    IRenderBackend* m_pActiveBackend = nullptr;
    bool m_HeadlessPacing = false;
    bool m_HasPreservedFrame = false;
    double m_PreservedX = 0, m_PreservedY = 0, m_PreservedW = 0, m_PreservedH = 0;
};
}  // namespace My

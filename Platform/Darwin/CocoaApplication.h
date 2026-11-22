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

   protected:
    NSWindow* m_pWindow = nullptr;
};
}  // namespace My

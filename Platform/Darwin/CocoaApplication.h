#include "BaseApplication.hpp"
#include "portable.hpp"

OBJC_CLASS(NSWindow);

namespace My {
class CocoaApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;

    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override;

    void CreateMainWindow() override;

    void GetFramebufferSize(uint32_t& width, uint32_t& height) override;

   protected:
    int InitializeGuiPlatform() override;
    void FinalizeGuiPlatform() override;
    void FinalizePlatformResources() override;
    NSWindow* m_pWindow = nullptr;
};
}  // namespace My

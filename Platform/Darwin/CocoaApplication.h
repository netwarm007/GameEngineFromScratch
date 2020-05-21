#include "BaseApplication.hpp"
#include "portable.hpp"

OBJC_CLASS(NSWindow);

namespace My {
class CocoaApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;

    void Finalize() override;
    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override;

   protected:
    void CreateMainWindow() override;

   protected:
    NSWindow* m_pWindow;
};
}  // namespace My

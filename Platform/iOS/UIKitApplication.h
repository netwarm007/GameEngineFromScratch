#include "BaseApplication.hpp"
#include "portable.hpp"

OBJC_CLASS(UIWindow);

namespace My {
class UIKitApplication : public BaseApplication {
   public:
    using BaseApplication::BaseApplication;

    void Finalize() override;
    // One cycle of the main loop
    void Tick() override;

    void* GetMainWindowHandler() override;

   protected:
    void CreateMainWindow() override;

   protected:
    UIWindow* m_pWindow;
};
}  // namespace My

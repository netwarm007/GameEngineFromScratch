#ifdef __OBJC__
#include <Cocoa/Cocoa.h>
#endif
#include "BaseApplication.hpp"

namespace My {
    class CocoaApplication : public BaseApplication
    {
    public:
        using BaseApplication::BaseApplication;

        void Finalize() override;
        // One cycle of the main loop
        void Tick() override;

        void* GetMainWindowHandler() override;
        void CreateMainWindow() override;

    protected:
#ifdef __OBJC__
        NSWindow* m_pWindow;
#endif
    };
}


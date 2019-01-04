#ifdef __OBJC__
#include <Cocoa/Cocoa.h>
#endif
#include "BaseApplication.hpp"

namespace My {
    class CocoaApplication : public BaseApplication
    {
    public:
        CocoaApplication(GfxConfiguration& config)
            : BaseApplication(config) {};

        int Initialize() override;
        void Finalize() override;
        // One cycle of the main loop
        void Tick() override;

        void* GetMainWindowHandler() override;

    protected:
        void CreateWindow();

    protected:
#ifdef __OBJC__
        NSWindow* m_pWindow;
#endif
    };
}


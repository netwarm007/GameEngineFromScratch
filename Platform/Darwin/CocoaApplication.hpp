#include "BaseApplication.hpp"
#include <Cocoa/Cocoa.h>

namespace My {
    class CocoaApplication : public BaseApplication
    {
    public:
        CocoaApplication(GfxConfiguration& config)
            : BaseApplication(config) {};

        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();

    protected:
        NSWindow* m_pWindow;
    };
}


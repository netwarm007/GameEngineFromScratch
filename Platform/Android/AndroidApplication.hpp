#pragma once
#include "BaseApplication.hpp"

namespace My {
    class AndroidApplication : public BaseApplication
    {
    public:
        AndroidApplication(GfxConfiguration& cfg);
        virtual int Initialize();
        virtual void Finalize();
        // One cycle of the main loop
        virtual void Tick();
    };
}


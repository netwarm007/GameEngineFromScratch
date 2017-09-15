#pragma once
#include "XcbApplication.hpp"

namespace My {
    class OpenGLApplication : public XcbApplication {
    public:
        OpenGLApplication(GfxConfiguration& config)
            : XcbApplication(config) {};

        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

    private:
    };
}

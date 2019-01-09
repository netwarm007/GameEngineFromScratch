#pragma once
#include "SdlApplication.hpp"

namespace My {
    class OpenGLApplication : public SdlApplication {
    public:
        OpenGLApplication(GfxConfiguration& config)
            : SdlApplication(config) {};

        virtual void Tick();

        void CreateMainWindow() override;
    };
}

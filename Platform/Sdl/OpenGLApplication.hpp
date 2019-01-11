#pragma once
#include "SdlApplication.hpp"

namespace My {
    class OpenGLApplication : public SdlApplication {
    public:
        OpenGLApplication(GfxConfiguration& config)
            : SdlApplication(config) {};

        void Tick() override;

        void CreateMainWindow() override;

    private:
        SDL_GLContext m_hContext;
    };
}

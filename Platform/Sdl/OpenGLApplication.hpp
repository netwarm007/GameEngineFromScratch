#pragma once
#include "SdlApplication.hpp"

namespace My {
    class OpenGLApplication : public SdlApplication {
    public:
        using SdlApplication::SdlApplication;

        void Tick() override;

        void CreateMainWindow() override;

    private:
        SDL_GLContext m_hContext;
    };
}

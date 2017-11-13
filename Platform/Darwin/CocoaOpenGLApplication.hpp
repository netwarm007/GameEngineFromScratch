#pragma once
#include "CocoaApplication.hpp"
#include "GLView.h"

namespace My {
    class CocoaOpenGLApplication : public CocoaApplication {
    public:
        CocoaOpenGLApplication(GfxConfiguration& config)
            : CocoaApplication(config) {};

        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

    protected:
        virtual void OnDraw();

    private:
        GLView* m_pGlView;
    };
}


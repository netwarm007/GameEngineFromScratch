#pragma once
#include "CocoaApplication.h"
#ifdef __OBJC__
#include "GLView.h"
#endif

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
#ifdef __OBJC__
        GLView* m_pGlView;
#endif
    };
}


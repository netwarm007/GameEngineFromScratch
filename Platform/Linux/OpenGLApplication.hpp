#pragma once
#include "glad/glad_glx.h"
#include "XcbApplication.hpp"

namespace My {
    class OpenGLApplication : public XcbApplication {
    public:
        using XcbApplication::XcbApplication;

        int Initialize() override;
        void Tick() override;

        void CreateMainWindow() override;

    private:
        Display *m_pDisplay;
        GLXContext m_Context;
        GLXDrawable m_Drawable;
        GLXFBConfig fb_config;
        XVisualInfo *vi;
    };
}

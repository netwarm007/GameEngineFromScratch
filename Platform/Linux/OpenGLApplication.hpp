#pragma once

#include "glad/glad_glx.h"

#include "XcbApplication.hpp"

namespace My {
class OpenGLApplication : public XcbApplication {
   public:
    using XcbApplication::XcbApplication;

    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void CreateMainWindow() override;

   private:
    GLXContext m_Context;
    GLXWindow m_GlxWindow;
    GLXFBConfig fb_config;
};
}  // namespace My

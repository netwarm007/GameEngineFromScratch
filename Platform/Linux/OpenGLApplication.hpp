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
    void FinalizeGraphicsResources() override;
    GLXContext m_Context = nullptr;
    GLXWindow m_GlxWindow = 0;
    GLXFBConfig fb_config;
};
}  // namespace My

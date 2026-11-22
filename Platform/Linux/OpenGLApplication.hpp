#pragma once

#include "IRenderBackend.hpp"
#include "glad/glad_glx.h"

#include "XcbApplication.hpp"

namespace My {
class OpenGLApplication : public XcbApplication, public IRenderBackend {
   public:
    using XcbApplication::XcbApplication;

    void Tick() override;

    void CreateMainWindow() override;

    int InitializeGuiPlatform() override {
        return XcbApplication::InitializeGuiPlatform();
    }
    void FinalizeGuiPlatform() override { XcbApplication::FinalizeGuiPlatform(); }

   private:
    int InitializeGraphicsResources() override;
    void WaitForGraphicsIdle() override;
    void FinalizeGraphicsResources() override;
    void PresentFrame() override;
    GLXContext m_Context = nullptr;
    GLXWindow m_GlxWindow = 0;
    GLXFBConfig fb_config;
};
}  // namespace My

#include "OpenGLApplication.hpp"

namespace My {
extern GfxConfiguration config;
IApplication* g_pApp =
    static_cast<IApplication*>(new OpenGLApplication(config));
}  // namespace My

#include "CocoaMetalApplication.h"

namespace My {
extern GfxConfiguration config;
IApplication* g_pApp =
    static_cast<IApplication*>(new CocoaMetalApplication(config));
}  // namespace My

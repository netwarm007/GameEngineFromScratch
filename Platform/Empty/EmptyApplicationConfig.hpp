#include "EmptyApplication.hpp"

namespace My {
extern GfxConfiguration config;
IApplication* g_pApp = static_cast<IApplication*>(new BaseApplication(config));
}  // namespace My

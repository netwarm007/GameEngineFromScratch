#include "D3d12Application.hpp"

namespace My {
extern GfxConfiguration config;
IApplication* g_pApp = static_cast<IApplication*>(new D3d12Application(config));
}  // namespace My

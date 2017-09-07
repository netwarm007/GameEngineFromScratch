#include "BaseApplication.hpp"

namespace My {
    GfxConfiguration config;
	BaseApplication g_App(config);
	IApplication* g_pApp = &g_App;
}


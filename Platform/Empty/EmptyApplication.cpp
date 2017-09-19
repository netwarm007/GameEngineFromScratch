#include "BaseApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"

namespace My {
    GfxConfiguration config;
	IApplication*    g_pApp             = static_cast<IApplication*>(new BaseApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
}


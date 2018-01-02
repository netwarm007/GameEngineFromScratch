#include "AndroidApplication.hpp"

namespace My {
    GfxConfiguration config;
        IApplication*    g_pApp             = static_cast<IApplication*>(new AndroidApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
}

using namespace My;
using namespace std;

AndroidApplication::AndroidApplication(GfxConfiguration& cfg) 
        : BaseApplication(cfg)
{
}

int AndroidApplication::Initialize()
{
        return 0;
}

void AndroidApplication::Finalize()
{
}

void AndroidApplication::Tick()
{
}


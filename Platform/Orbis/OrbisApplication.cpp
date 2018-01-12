#include "OrbisApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"

namespace My {
    GfxConfiguration config;
	IApplication*    g_pApp             = static_cast<IApplication*>(new OrbisApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
}

using namespace My;
using namespace std;

int OrbisApplication::Initialize()
{
    int ret;
    ret = BaseApplication::Initialize();
    if (!ret)
    {
        g_pAssetLoader->AddSearchPath("/app0");
        g_pAssetLoader->AddSearchPath("/hostapp");
    }

    return ret;
}



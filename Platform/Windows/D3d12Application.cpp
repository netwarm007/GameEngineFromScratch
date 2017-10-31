#include "WindowsApplication.hpp"
#include "D3d/D3d12GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"
#include <tchar.h>

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 32, 0, 0, 960, 540, _T("Game Engine From Scratch (Windows)"));
	IApplication* g_pApp                = static_cast<IApplication*>(new WindowsApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new D3d12GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
}


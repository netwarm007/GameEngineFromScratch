#pragma once
#include "IApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"

using namespace My;

namespace My {
	extern IApplication*    g_pApp;
    extern MemoryManager*   g_pMemoryManager;
    extern GraphicsManager* g_pGraphicsManager;
    extern AssetLoader*     g_pAssetLoader;
    extern SceneManager*    g_pSceneManager;
}


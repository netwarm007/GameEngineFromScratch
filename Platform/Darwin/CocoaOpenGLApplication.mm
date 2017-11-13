#include <stdio.h>
#include <climits>
#include <cstring>
#include "CocoaOpenGLApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"

using namespace My;

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "Game Engine From Scratch (MacOS Cocoa)");
    IApplication* g_pApp                = static_cast<IApplication*>(new CocoaOpenGLApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
}

int CocoaOpenGLApplication::Initialize()
{
    int result = 0;

    result = CocoaApplication::Initialize();

    return result;
}

void CocoaOpenGLApplication::Finalize()
{
    CocoaApplication::Finalize();
}

void CocoaOpenGLApplication::Tick()
{
    CocoaApplication::Tick();
}

void CocoaOpenGLApplication::OnDraw()
{
}


#include "CocoaApplication.h"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "Game Engine From Scratch (MacOS Cocoa)");
    IApplication* g_pApp                = static_cast<IApplication*>(new CocoaApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new GraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager);
    PhysicsManager*  g_pPhysicsManager  = static_cast<PhysicsManager*>(new PhysicsManager);
    GameLogic*       g_pGameLogic       = static_cast<GameLogic*>(new GameLogic);
}


#include "config.h"
#if defined(APPLE) 
#include "CocoaOpenGLApplication.h"
#elif defined(WIN32)
#include "OpenGLApplication.hpp"
#endif
#include "OpenGL/OpenGLGraphicsManager.hpp"
#include "BilliardGameLogic.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 1920, 1080, "Billiard Game");
#if defined(APPLE) 
    IApplication* g_pApp                = static_cast<IApplication*>(new CocoaOpenGLApplication(config));
#elif defined(WIN32)
    IApplication* g_pApp                = static_cast<IApplication*>(new OpenGLApplication(config));
#elif defined(ANDROID)
    IApplication* g_pApp                = static_cast<IApplication*>(new AndroidApplication(config));
#elif defined(UNIX)
    IApplication* g_pApp                = static_cast<IApplication*>(new OpenGLApplication(config));
#else
    IApplication* g_pApp                = static_cast<IApplication*>(new EmptyApplication(config));
#endif
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager);
    PhysicsManager*  g_pPhysicsManager  = static_cast<PhysicsManager*>(new PhysicsManager);
    GameLogic*       g_pGameLogic       = static_cast<GameLogic*>(new BilliardGameLogic);
}

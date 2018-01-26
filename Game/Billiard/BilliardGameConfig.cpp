#include "config.h"
#if defined(APPLE) 
#include "CocoaOpenGLApplication.h"
#include "OpenGL/OpenGLGraphicsManager.hpp"
#elif defined(WIN32)
#include "OpenGLApplication.hpp"
#include "OpenGL/OpenGLGraphicsManager.hpp"
#elif defined(ANDROID)
#include "AndroidApplication.hpp"
#include "OpenGL/OpenGLESGraphicsManager.hpp"
#elif defined(UNIX)
#include "OpenGLApplication.hpp"
#include "OpenGL/OpenGLGraphicsManager.hpp"
#else
#include "EmptyApplication.hpp"
#endif
#include "BilliardGameLogic.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 1920, 1080, "Billiard Game");
#if defined(APPLE) 
    IApplication* g_pApp                = static_cast<IApplication*>(new CocoaOpenGLApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
#elif defined(WIN32)
    IApplication* g_pApp                = static_cast<IApplication*>(new OpenGLApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
#elif defined(ANDROID)
    IApplication* g_pApp                = static_cast<IApplication*>(new AndroidApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLESGraphicsManager);
#elif defined(UNIX)
    IApplication* g_pApp                = static_cast<IApplication*>(new OpenGLApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
#else
    IApplication* g_pApp                = static_cast<IApplication*>(new EmptyApplication(config));
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new OpenGLGraphicsManager);
#endif
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager);
    PhysicsManager*  g_pPhysicsManager  = static_cast<PhysicsManager*>(new PhysicsManager);
    GameLogic*       g_pGameLogic       = static_cast<GameLogic*>(new BilliardGameLogic);
}

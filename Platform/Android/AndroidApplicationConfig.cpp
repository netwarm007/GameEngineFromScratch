#include "OpenGLESApplication.hpp"
#include "AndroidAssetLoader.hpp"
#include "OpenGL/OpenGLESConfig.hpp"

namespace My {
    extern GfxConfiguration config;
    IApplication*    g_pApp             = static_cast<IApplication*>(new OpenGLESApplication(config));
    MemoryManager*   g_pMemoryManager   = static_cast<MemoryManager*>(new MemoryManager());
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AndroidAssetLoader());
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager());
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager());
    AnimationManager* g_pAnimationManager = static_cast<AnimationManager*>(new AnimationManager);
#ifdef DEBUG
    DebugManager*    g_pDebugManager    = static_cast<DebugManager*>(new DebugManager);
#endif
}


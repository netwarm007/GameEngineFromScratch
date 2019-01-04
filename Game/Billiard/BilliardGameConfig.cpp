#include "portable.hpp"
#include "GfxConfiguration.hpp"

#ifdef OS_MACOS
#include "CocoaMetalApplicationConfig.hpp"
#include "RHI/Metal/MetalConfig.hpp"
#else
#include "OpenGLApplicationConfig.hpp"
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif


#include "BilliardGameLogic.hpp"
#include "Physics/Bullet/BulletPhysicsManager.hpp"
#include "Framework/Common/MemoryManager.hpp"
#include "Framework/Common/AssetLoader.hpp"
#include "Framework/Common/SceneManager.hpp"
#include "Framework/Common/InputManager.hpp"
#include "Framework/Common/AnimationManager.hpp"
#include "Framework/Common/DebugManager.hpp"

namespace My {
    GfxConfiguration config(8, 8, 8, 8, 24, 8, 0, 960, 540, "Billiard Game");
    IGameLogic*       g_pGameLogic       = static_cast<IGameLogic*>(new BilliardGameLogic);
    IPhysicsManager*  g_pPhysicsManager  = static_cast<IPhysicsManager*>(new BulletPhysicsManager);
    IMemoryManager*   g_pMemoryManager   = static_cast<IMemoryManager*>(new MemoryManager);
    AssetLoader*     g_pAssetLoader     = static_cast<AssetLoader*>(new AssetLoader);
    SceneManager*    g_pSceneManager    = static_cast<SceneManager*>(new SceneManager);
    InputManager*    g_pInputManager    = static_cast<InputManager*>(new InputManager);
    AnimationManager* g_pAnimationManager = static_cast<AnimationManager*>(new AnimationManager);
#ifdef DEBUG
    DebugManager*    g_pDebugManager    = static_cast<DebugManager*>(new DebugManager);
#endif
}


#include "GfxConfiguration.hpp"
#include "portable.hpp"
#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#else
#include "OpenGLApplication.hpp"
#endif
#if defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif
#include "BilliardGameLogic.hpp"
#include "Framework/Common/AnimationManager.hpp"
#include "Framework/Common/AssetLoader.hpp"
#include "Framework/Common/DebugManager.hpp"
#include "Framework/Common/InputManager.hpp"
#include "Framework/Common/MemoryManager.hpp"
#include "Framework/Common/SceneManager.hpp"
#include "Physics/Bullet/BulletPhysicsManager.hpp"

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 960, 540, "Billiard Game");
#if defined(OS_MACOS)
IApplication* g_pApp =
    static_cast<IApplication*>(new CocoaMetalApplication(config));
#else
IApplication* g_pApp =
    static_cast<IApplication*>(new OpenGLApplication(config));
#endif
IGameLogic* g_pGameLogic = static_cast<IGameLogic*>(new BilliardGameLogic);
IPhysicsManager* g_pPhysicsManager =
    static_cast<IPhysicsManager*>(new BulletPhysicsManager);
IMemoryManager* g_pMemoryManager =
    static_cast<IMemoryManager*>(new MemoryManager);
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new AssetLoader);
SceneManager* g_pSceneManager = static_cast<SceneManager*>(new SceneManager);
InputManager* g_pInputManager = static_cast<InputManager*>(new InputManager);
AnimationManager* g_pAnimationManager =
    static_cast<AnimationManager*>(new AnimationManager);
#ifdef DEBUG
DebugManager* g_pDebugManager = static_cast<DebugManager*>(new DebugManager);
#endif
}  // namespace My

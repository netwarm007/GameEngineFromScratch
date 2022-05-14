#include "GfxConfiguration.hpp"
#include "portable.hpp"
#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#include "BundleAssetLoader.h"
#elif defined(OS_IOS)
#include "UIKitApplication.h"
#else
#include "OpenGLApplication.hpp"
#endif
#if defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS) || defined(OS_IOS)
#include "RHI/Metal/MetalConfig.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif
#include "BilliardGameLogic.hpp"
#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "SceneManager.hpp"
#include "Bullet/BulletPhysicsManager.hpp"

namespace My {
GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 960, 540, "Billiard Game");
#if defined(OS_MACOS)
IApplication* g_pApp =
    static_cast<IApplication*>(new CocoaMetalApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new BundleAssetLoader);
#elif defined(OS_IOS)
IApplication* g_pApp =
    static_cast<IApplication*>(new UIKitApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new AssetLoader);
#elif defined(OS_WINDOWS)
// IApplication* g_pApp = static_cast<IApplication*>(new
// D3d12Application(config));
IApplication* g_pApp =
    static_cast<IApplication*>(new OpenGLApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new AssetLoader);
#else
IApplication* g_pApp =
    static_cast<IApplication*>(new OpenGLApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new AssetLoader);
#endif
IGameLogic* g_pGameLogic = static_cast<IGameLogic*>(new BilliardGameLogic);
IPhysicsManager* g_pPhysicsManager =
    static_cast<IPhysicsManager*>(new BulletPhysicsManager);
IMemoryManager* g_pMemoryManager =
    static_cast<IMemoryManager*>(new MemoryManager);
SceneManager* g_pSceneManager = static_cast<SceneManager*>(new SceneManager);
InputManager* g_pInputManager = static_cast<InputManager*>(new InputManager);
AnimationManager* g_pAnimationManager =
    static_cast<AnimationManager*>(new AnimationManager);
#ifdef DEBUG
DebugManager* g_pDebugManager = static_cast<DebugManager*>(new DebugManager);
#endif
}  // namespace My

#include "GfxConfiguration.hpp"
#include "config.h"
#include "BaseApplication.hpp"

#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "My/MyPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "GameLogic.hpp"

#if defined(OS_WEBASSEMBLY)
#include "Platform/Sdl/OpenGLApplication.hpp"
#elif defined(OS_MACOS)
#include "CocoaMetalApplication.h"
#include "BundleAssetLoader.h"
#elif defined(OS_WINDOWS)
//#include "D3d12Application.hpp"
#include "OpenGLApplication.hpp"
#else
#include "OpenGLApplication.hpp"
#endif

#if defined(OS_ANDROID) || defined(OS_WEBASSEMBLY)
#include "RHI/OpenGL/OpenGLESConfig.hpp"
#elif defined(OS_MACOS)
#include "RHI/Metal/MetalConfig.hpp"
#elif defined(OS_WINDOWS)
//#include "RHI/D3d/D3d12Config.hpp"
#include "RHI/OpenGL/OpenGLConfig.hpp"
#else
#include "RHI/OpenGL/OpenGLConfig.hpp"
#endif

namespace My {
extern GfxConfiguration config;
extern GameLogic* g_pGameLogic;
extern PhysicsManager* g_pPhysicsManager;
#if defined(OS_MACOS)
BaseApplication* g_pApp =
    static_cast<BaseApplication*>(new CocoaMetalApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new BundleAssetLoader);
#elif defined(OS_WINDOWS)
// BaseApplication* g_pApp = static_cast<BaseApplication*>(new
// D3d12Application(config));
BaseApplication* g_pApp =
    static_cast<BaseApplication*>(new OpenGLApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new AssetLoader);
#else
BaseApplication* g_pApp =
    static_cast<BaseApplication*>(new OpenGLApplication(config));
AssetLoader* g_pAssetLoader = static_cast<AssetLoader*>(new AssetLoader);
#endif
MemoryManager* g_pMemoryManager =
    static_cast<MemoryManager*>(new MemoryManager);
SceneManager* g_pSceneManager = static_cast<SceneManager*>(new SceneManager);
InputManager* g_pInputManager = static_cast<InputManager*>(new InputManager);
AnimationManager* g_pAnimationManager =
    static_cast<AnimationManager*>(new AnimationManager);
#ifdef DEBUG
DebugManager* g_pDebugManager = static_cast<DebugManager*>(new DebugManager);
#endif
}  // namespace My

#if defined(OS_WEBASSEMBLY)
#include <emscripten.h>

#include <functional>

std::function<void()> loop;
void main_loop() { loop(); }
#endif  // defined(OS_WEBASSEMBLY)

using namespace My;

int main(int argc, char** argv) {
    int ret;

    g_pApp->SetCommandLineParameters(argc, argv);

    g_pApp->RegisterManagerModule(g_pAnimationManager, g_pAnimationManager);
    g_pApp->RegisterManagerModule(g_pAssetLoader, g_pAssetLoader);
    g_pApp->RegisterManagerModule(g_pDebugManager, g_pDebugManager);
    g_pApp->RegisterManagerModule(g_pGraphicsManager, g_pGraphicsManager);
    g_pApp->RegisterManagerModule(g_pInputManager, g_pInputManager);
    g_pApp->RegisterManagerModule(g_pMemoryManager, g_pMemoryManager);
    g_pApp->RegisterManagerModule(g_pPhysicsManager, g_pPhysicsManager);
    g_pApp->RegisterManagerModule(g_pPipelineStateManager, g_pPipelineStateManager);
    g_pApp->RegisterManagerModule(g_pSceneManager, g_pSceneManager);
    g_pApp->RegisterManagerModule(g_pGameLogic, g_pGameLogic);

    g_pApp->CreateMainWindow();

    g_pApp->Initialize();

    // Drive the modules ahead
#if defined(OS_WEBASSEMBLY)
    auto main_loop = [] { g_pApp->Tick(); };

    emscripten_set_main_loop(main_loop, 0, true);
#else
    while (!g_pApp->IsQuit()) {
        g_pApp->Tick();
    }
#endif

#if !defined(OS_WEBASSEMBLY)
    // Finalize App
    g_pApp->Finalize();
#endif

    delete g_pAnimationManager;
    delete g_pAssetLoader;
    delete g_pDebugManager;
    delete g_pGameLogic;
    delete g_pGraphicsManager;
    delete g_pInputManager;
    delete g_pMemoryManager;
    delete g_pPhysicsManager;
    delete g_pPipelineStateManager;
    delete g_pSceneManager;

    return EXIT_SUCCESS;
}

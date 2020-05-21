#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "DebugManager.hpp"
#include "GraphicsManager.hpp"
#include "IApplication.hpp"
#include "IGameLogic.hpp"
#include "IPhysicsManager.hpp"
#include "IPipelineStateManager.hpp"
#include "InputManager.hpp"
#include "MemoryManager.hpp"
#include "SceneManager.hpp"
#include "portable.hpp"
#if defined(OS_WEBASSEMBLY)
#include <emscripten.h>

#include <functional>

std::function<void()> loop;
void main_loop() { loop(); }
#endif  // defined(OS_WEBASSEMBLY)

using namespace My;
using namespace std;

int main(int argc, char** argv) {
    int ret;

    g_pApp->SetCommandLineParameters(argc, argv);

    vector<IRuntimeModule*> run_time_modules;
    run_time_modules.push_back(g_pApp);
    run_time_modules.push_back(g_pMemoryManager);
    run_time_modules.push_back(g_pAssetLoader);
    run_time_modules.push_back(g_pSceneManager);
    run_time_modules.push_back(g_pGraphicsManager);
    run_time_modules.push_back(g_pPipelineStateManager);
    run_time_modules.push_back(g_pInputManager);
    run_time_modules.push_back(g_pPhysicsManager);
    run_time_modules.push_back(g_pAnimationManager);
    run_time_modules.push_back(g_pGameLogic);
#ifdef DEBUG
    run_time_modules.push_back(g_pDebugManager);
#endif

    // Initialize Runtime Modules
    for (auto& module : run_time_modules) {
        if ((ret = module->Initialize()) != 0) {
            cerr << "Failed. err = " << ret;
            return EXIT_FAILURE;
        }
    }

    // Drive the modules ahead
#if defined(OS_WEBASSEMBLY)
    loop = [&] {
        for (auto& module : run_time_modules) {
            module->Tick();
        }
    };

    emscripten_set_main_loop(main_loop, 0, true);
#else
    while (!g_pApp->IsQuit()) {
        for (auto& module : run_time_modules) {
            module->Tick();
        }
    }
#endif

#if !defined(OS_WEBASSEMBLY)
    // Finalize Runtime Modules
    for (auto& module : run_time_modules) {
        module->Finalize();
    }

    // Finalize App
    g_pApp->Finalize();
#endif

    return EXIT_SUCCESS;
}

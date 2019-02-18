#include <cstdio>
#include <chrono>
#include <thread>
#include "portable.hpp"
#include "IApplication.hpp"
#include "IShaderManager.hpp"
#include "IPhysicsManager.hpp"
#include "IGameLogic.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"
#include "InputManager.hpp"
#include "DebugManager.hpp"
#include "AnimationManager.hpp"
#if defined(OS_WEBASSEMBLY)
#include <functional>
#include <emscripten.h>

std::function<void()> loop;
void main_loop() { loop(); }
#endif // defined(OS_WEBASSEMBLY)

using namespace My;
using namespace std;

int main(int argc, char** argv) {
	int ret;

    g_pApp->SetCommandLineParameters(argc, argv);

	if ((ret = g_pApp->Initialize()) != 0) {
		printf("App Initialize failed, will exit now.");
		return ret;
	}

	// create the main window
	g_pApp->CreateMainWindow();

	// Initialize Runtime Modules
	if ((ret = g_pMemoryManager->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

	if ((ret = g_pAssetLoader->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

	if ((ret = g_pSceneManager->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

	if ((ret = g_pGraphicsManager->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

	if ((ret = g_pShaderManager->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

	if ((ret = g_pInputManager->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

	if ((ret = g_pPhysicsManager->Initialize()) != 0) {
        cerr << "Failed. err = " << ret;
		return ret;
	}

    if ((ret = g_pAnimationManager->Initialize()) != 0) {
        cerr << "Failed. err =" << ret;
        return ret;
    }

    if ((ret = g_pGameLogic->Initialize()) != 0) {
        cerr << "Failed. err =" << ret;
        return ret;
    }

#ifdef DEBUG
    if ((ret = g_pDebugManager->Initialize()) != 0) {
        cerr << "Failed. err =" << ret;
        return ret;
    }
#endif

#if defined(OS_WEBASSEMBLY)
    loop = [&]
    {
		g_pApp->Tick();
		g_pMemoryManager->Tick();
		g_pAssetLoader->Tick();
		g_pSceneManager->Tick();
		g_pInputManager->Tick();
		g_pPhysicsManager->Tick();
		g_pAnimationManager->Tick();
		g_pShaderManager->Tick();
		g_pGameLogic->Tick();
		g_pGraphicsManager->Tick();
	#ifdef DEBUG
		g_pDebugManager->Tick();
	#endif
    };

    emscripten_set_main_loop(main_loop, 0, true);
#else
	while (!g_pApp->IsQuit()) {
		g_pApp->Tick();
		g_pMemoryManager->Tick();
		g_pAssetLoader->Tick();
		g_pSceneManager->Tick();
		g_pInputManager->Tick();
		g_pPhysicsManager->Tick();
		g_pAnimationManager->Tick();
		g_pShaderManager->Tick();
		g_pGameLogic->Tick();
		g_pGraphicsManager->Tick();
	#ifdef DEBUG
		g_pDebugManager->Tick();
	#endif
	}
#endif

#if !defined(OS_WEBASSEMBLY)
	// Finalize Runtime Modules
#ifdef DEBUG
    g_pDebugManager->Finalize();
#endif
    g_pGameLogic->Finalize();
    g_pAnimationManager->Finalize();
    g_pPhysicsManager->Finalize();
    g_pInputManager->Finalize();
    g_pShaderManager->Finalize();
    g_pGraphicsManager->Finalize();
    g_pSceneManager->Finalize();
    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

	// Finalize App
	g_pApp->Finalize();
#endif

	return 0;
}


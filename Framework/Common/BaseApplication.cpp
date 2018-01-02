#include "BaseApplication.hpp"
#include <iostream>

using namespace My;
using namespace std;

bool BaseApplication::m_bQuit = false;

BaseApplication::BaseApplication(GfxConfiguration& cfg)
    :m_Config(cfg)
{
}

// Parse command line, read configuration, initialize all sub modules
int BaseApplication::Initialize()
{
    int ret = 0;

    std::cout << m_Config;

	if ((ret = g_pMemoryManager->Initialize()) != 0) {
		printf("Memory Manager Initialize failed, will exit now.");
		return ret;
	}

	if ((ret = g_pAssetLoader->Initialize()) != 0) {
		printf("Asset Loader Initialize failed, will exit now.");
		return ret;
	}

	if ((ret = g_pSceneManager->Initialize()) != 0) {
		printf("Scene Manager Initialize failed, will exit now.");
		return ret;
	}

    string scene_file_name = "Scene/test.ogex";
    if (m_nArgC > 1) {
        scene_file_name = m_ppArgV[1];
    }

    if ((ret = g_pSceneManager->LoadScene(scene_file_name.c_str())) != 0) {
        printf("Unable to load scene: %s\n", scene_file_name.c_str());
        return ret;
    }

	if ((ret = g_pGraphicsManager->Initialize()) != 0) {
		printf("Graphics Manager Initialize failed, will exit now.");
		return ret;
	}

	return ret;
}


// Finalize all sub modules and clean up all runtime temporary files.
void BaseApplication::Finalize()
{
    g_pGraphicsManager->Finalize();
    g_pSceneManager->Finalize();
    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();
}


// One cycle of the main loop
void BaseApplication::Tick()
{
    g_pMemoryManager->Tick();
    g_pAssetLoader->Tick();
    g_pSceneManager->Tick();
    g_pGraphicsManager->Tick();
}

void BaseApplication::SetCommandLineParameters(int argc, char** argv)
{
    m_nArgC = argc;
    m_ppArgV = argv;
}

bool BaseApplication::IsQuit()
{
	return m_bQuit;
}


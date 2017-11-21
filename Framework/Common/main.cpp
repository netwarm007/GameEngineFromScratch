#include <cstdio>
#include <unistd.h>
#include "EmptyApplication.hpp"

using namespace My;

int main(int argc, char** argv) {
	int ret;

	if ((ret = g_pApp->Initialize()) != 0) {
		printf("App Initialize failed, will exit now.");
		return ret;
	}

	if ((ret = g_pMemoryManager->Initialize()) != 0) {
		printf("Memory Manager Initialize failed, will exit now.");
		return ret;
	}

	if ((ret = g_pGraphicsManager->Initialize()) != 0) {
		printf("Graphics Manager Initialize failed, will exit now.");
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

	while (!g_pApp->IsQuit()) {
		g_pApp->Tick();
        g_pMemoryManager->Tick();
        g_pGraphicsManager->Tick();
        g_pAssetLoader->Tick();
        g_pSceneManager->Tick();
        usleep(10000);
	}

    g_pSceneManager->Finalize();
    g_pAssetLoader->Finalize();
    g_pGraphicsManager->Finalize();
    g_pMemoryManager->Finalize();
	g_pApp->Finalize();

	return 0;
}


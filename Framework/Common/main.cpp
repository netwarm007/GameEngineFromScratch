#include <stdio.h>
#include "IApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"

using namespace My;

namespace My {
	extern IApplication*    g_pApp;
    extern MemoryManager*   g_pMemoryManager;
    extern GraphicsManager* g_pGraphicsManager;
}

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

	while (!g_pApp->IsQuit()) {
		g_pApp->Tick();
        g_pMemoryManager->Tick();
        g_pGraphicsManager->Tick();
	}

    g_pGraphicsManager->Finalize();
    g_pMemoryManager->Finalize();
	g_pApp->Finalize();

	return 0;
}


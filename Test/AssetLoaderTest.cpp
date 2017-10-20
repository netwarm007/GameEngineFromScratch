#include <iostream>
#include <string>
#include "AssetLoader.hpp"
#include "MemoryManager.hpp"

using namespace std;
using namespace My;

namespace My {
    MemoryManager* g_pMemoryManager = new MemoryManager();
}

int main(int , char** )
{
    g_pMemoryManager->Initialize();

    AssetLoader asset_loader;
#ifdef __ORBIS__
    asset_loader.AddSearchPath("/app0");
#endif
    string shader_pgm = asset_loader.SyncOpenAndReadTextFileToString("Shaders/copy.vs");

    cout << shader_pgm;

    g_pMemoryManager->Finalize();

    delete g_pMemoryManager;

    return 0;
}


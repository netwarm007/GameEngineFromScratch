#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "MemoryManager.hpp"

using namespace std;
using namespace My;

namespace My {
IMemoryManager* g_pMemoryManager = new MemoryManager();
AssetLoader* g_pAssetLoader = new AssetLoader();
}  // namespace My

int main(int, char**) {
    g_pMemoryManager->Initialize();
    g_pAssetLoader->Initialize();

#ifdef __ORBIS__
    g_pAssetLoader->AddSearchPath("/app0");
#endif
    string shader_pgm = g_pAssetLoader->SyncOpenAndReadTextFileToString(
        "Shaders/HLSL/basic.vert.hlsl");

    cout << shader_pgm;

    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    return 0;
}

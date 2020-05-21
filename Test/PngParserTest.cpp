#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "MemoryManager.hpp"
#include "PNG.hpp"

using namespace std;
using namespace My;

namespace My {
IMemoryManager* g_pMemoryManager = new MemoryManager();
AssetLoader* g_pAssetLoader = new AssetLoader();
}  // namespace My

int main(int argc, const char** argv) {
    g_pMemoryManager->Initialize();
    g_pAssetLoader->Initialize();

#ifdef __ORBIS__
    g_pAssetLoader->AddSearchPath("/app0");
#endif

    {
        Buffer buf;
        if (argc >= 2) {
            buf = g_pAssetLoader->SyncOpenAndReadBinary(argv[1]);
        } else {
            buf = g_pAssetLoader->SyncOpenAndReadBinary("Textures/eye.png");
        }

        PngParser png_parser;

        Image image = png_parser.Parse(buf);

        cout << image;
    }

    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    return 0;
}

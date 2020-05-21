#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "HDR.hpp"
#include "MemoryManager.hpp"

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
            buf = g_pAssetLoader->SyncOpenAndReadBinary(
                "Textures/hdr/PaperMill_E_3k.hdr");
        }

        HdrParser hdr_parser;

        Image image = hdr_parser.Parse(buf);

        cout << image;
    }

    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    return 0;
}

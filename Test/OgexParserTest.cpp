#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "MemoryManager.hpp"
#include "OGEX.hpp"

using namespace My;
using namespace std;

namespace My {
IMemoryManager* g_pMemoryManager = new MemoryManager();
AssetLoader* g_pAssetLoader = new AssetLoader();
}  // namespace My

template <typename Key, typename T>
static ostream& operator<<(ostream& out,
                           unordered_map<Key, shared_ptr<T>> map) {
    for (const auto& p : map) {
        if (auto ptr = p.second) out << *ptr << endl;
    }

    return out;
}

int main(int, char**) {
    g_pMemoryManager->Initialize();
    g_pAssetLoader->Initialize();

    string ogex_text =
        g_pAssetLoader->SyncOpenAndReadTextFileToString("Scene/splash.ogex");

    auto* ogex_parser = new OgexParser();
    {
        shared_ptr<Scene> pScene = ogex_parser->Parse(ogex_text);
    }  // note texture in the scene will be async loaded until process terminate
    delete ogex_parser;

    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    return 0;
}

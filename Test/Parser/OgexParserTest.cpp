#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "OGEX.hpp"

using namespace My;
using namespace std;

template <typename Key, typename T>
static ostream& operator<<(ostream& out,
                           unordered_map<Key, shared_ptr<T>> map) {
    for (const auto& p : map) {
        if (auto ptr = p.second) out << *ptr << endl;
    }

    return out;
}

int main(int, char**) {
    AssetLoader assetLoader;
    assetLoader.Initialize();

    string ogex_text =
        assetLoader.SyncOpenAndReadTextFileToString("Scene/splash.ogex");

    OgexParser ogexParser;
    {
        shared_ptr<Scene> pScene = ogexParser.Parse(ogex_text);
    }  // note texture in the scene will be async loaded until process terminate

    assetLoader.Finalize();

    return 0;
}

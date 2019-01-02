#include <iostream>
#include <string>
#include "AssetLoader.hpp"
#include "MemoryManager.hpp"
#include "OGEX.hpp"

using namespace My;
using namespace std;

namespace My {
    IMemoryManager* g_pMemoryManager = new MemoryManager();
    AssetLoader*   g_pAssetLoader   = new AssetLoader();
}

template<typename Key, typename T>
static ostream& operator<<(ostream& out, unordered_map<Key, shared_ptr<T>> map)
{
    for (auto p : map)
    {
        if (auto ptr = p.second)
            out << *ptr << endl;
    }

    return out;
}

int main(int , char** )
{
    g_pMemoryManager->Initialize();
    g_pAssetLoader->Initialize();

    string ogex_text = g_pAssetLoader->SyncOpenAndReadTextFileToString("Scene/splash.ogex");

    OgexParser* ogex_parser = new OgexParser ();
    unique_ptr<Scene> pScene = ogex_parser->Parse(ogex_text);
    delete ogex_parser;

    cout << "Dump of Scene Graph" << endl;
    cout << "---------------------------" << endl;
    cout << *pScene->SceneGraph << endl;

    cout << "Dump of Cameras" << endl;
    cout << "---------------------------" << endl;
    cout << pScene->Cameras << endl;

    cout << "Dump of Lights" << endl;
    cout << "---------------------------" << endl;
    cout << pScene->Lights  << endl;

    cout << "Dump of Geometries" << endl;
    cout << "---------------------------" << endl;
    cout << pScene->Geometries << endl;

    cout << "Dump of Materials" << endl;
    cout << "---------------------------" << endl;
    cout << pScene->Materials << endl;

    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    delete g_pAssetLoader;
    delete g_pMemoryManager;

    return 0;
}


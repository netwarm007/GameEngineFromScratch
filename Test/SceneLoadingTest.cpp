#include <iostream>
#include <string>
#include "AssetLoader.hpp"
#include "MemoryManager.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

namespace My {
    MemoryManager*  g_pMemoryManager = new MemoryManager();
    AssetLoader*    g_pAssetLoader   = new AssetLoader();
    SceneManager*   g_pSceneManager  = new SceneManager();
}

template<typename T>
static ostream& operator<<(ostream& out, forward_list<shared_ptr<T>> list)
{
    for (auto p : list)
    {
        out << *p << endl;
    }

    return out;
}

int main(int , char** )
{
    g_pMemoryManager->Initialize();
    g_pSceneManager->Initialize();
    g_pAssetLoader->Initialize();

    g_pSceneManager->LoadScene("Scene/Example.ogex");

    cout << "Dump of Cameras" << endl;
    cout << "---------------------------" << endl;
    weak_ptr<SceneObjectCamera> pCamera = g_pSceneManager->GetFirstCamera();
    while(auto pObj = pCamera.lock())
    {
        cout << *pObj << endl;
        pCamera = g_pSceneManager->GetNextCamera();
    }

    cout << "Dump of Lights" << endl;
    cout << "---------------------------" << endl;
    weak_ptr<SceneObjectLight> pLight = g_pSceneManager->GetFirstLight();
    while(auto pObj = pLight.lock())
    {
        cout << *pObj << endl;
        pLight = g_pSceneManager->GetNextLight();
    }

    cout << "Dump of Geometries" << endl;
    cout << "---------------------------" << endl;
    weak_ptr<SceneObjectGeometry> pGeometry = g_pSceneManager->GetFirstGeometry();
    while(auto pObj = pGeometry.lock())
    {
        cout << *pObj << endl;
        pGeometry = g_pSceneManager->GetNextGeometry();
    }

    cout << "Dump of Materials" << endl;
    cout << "---------------------------" << endl;
    weak_ptr<SceneObjectMaterial> pMaterial = g_pSceneManager->GetFirstMaterial();
    while(auto pObj = pMaterial.lock())
    {
        cout << *pObj << endl;
        pMaterial = g_pSceneManager->GetNextMaterial();
    }

    g_pSceneManager->Finalize();
    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    delete g_pSceneManager;
    delete g_pAssetLoader;
    delete g_pMemoryManager;

    return 0;
}


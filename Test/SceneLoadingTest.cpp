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
static ostream& operator<<(ostream& out, unordered_map<string, shared_ptr<T>> map)
{
    for (auto p : map)
    {
        out << *p.second << endl;
    }

    return out;
}

int main(int , char** )
{
    g_pMemoryManager->Initialize();
    g_pSceneManager->Initialize();
    g_pAssetLoader->Initialize();

    g_pSceneManager->LoadScene("Scene/aili.ogex");
    auto& scene = g_pSceneManager->GetSceneForRendering();

    cout << "Dump of Cameras" << endl;
    cout << "---------------------------" << endl;
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        weak_ptr<SceneObjectCamera> pCamera = scene.GetCamera(pCameraNode->GetSceneObjectRef());
        while(auto pObj = pCamera.lock())
        {
            cout << *pObj << endl;
            pCameraNode = scene.GetNextCameraNode();
            if(!pCameraNode) break;
            pCamera = scene.GetCamera(pCameraNode->GetSceneObjectRef());
        }
    }

    cout << "Dump of Lights" << endl;
    cout << "---------------------------" << endl;
    auto pLightNode = scene.GetFirstLightNode();
    if (pLightNode) {
        weak_ptr<SceneObjectLight> pLight = scene.GetLight(pLightNode->GetSceneObjectRef());
        while(auto pObj = pLight.lock())
        {
            cout << *pObj << endl;
            pLightNode = scene.GetNextLightNode();
            if(!pLightNode) break;
            pLight = scene.GetLight(pLightNode->GetSceneObjectRef());
        }
    }

    cout << "Dump of Geometries" << endl;
    cout << "---------------------------" << endl;
    auto pGeometryNode = scene.GetFirstGeometryNode();
    if (pGeometryNode) {
        weak_ptr<SceneObjectGeometry> pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
        while(auto pObj = pGeometry.lock())
        {
            cout << *pObj << endl;
            pGeometryNode = scene.GetNextGeometryNode();
            if(!pGeometryNode) break;
            pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
        }
    }

    cout << "Dump of Materials" << endl;
    cout << "---------------------------" << endl;
    weak_ptr<SceneObjectMaterial> pMaterial = scene.GetFirstMaterial();
    while(auto pObj = pMaterial.lock())
    {
        cout << *pObj << endl;
        pMaterial = scene.GetNextMaterial();
    }

    g_pSceneManager->Finalize();
    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    delete g_pSceneManager;
    delete g_pAssetLoader;
    delete g_pMemoryManager;

    return 0;
}


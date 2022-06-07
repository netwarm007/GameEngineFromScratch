#include <iostream>
#include <string>

#include "BaseApplication.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

template <typename T>
static ostream& operator<<(ostream& out,
                           unordered_map<string, shared_ptr<T>> map) {
    for (auto p : map) {
        out << *p.second << endl;
    }

    return out;
}

void loadScene(SceneManager& sceneManager, const char* scene_name) {
    sceneManager.LoadScene(scene_name);
    auto& scene = sceneManager.GetSceneForRendering();

    cout << "Dump of Geometries" << endl;
    cout << "---------------------------" << endl;
    for (const auto& _it : scene->GeometryNodes)
    {
        auto pGeometryNode = _it.second.lock();
        cout << *pGeometryNode << endl;
        if (pGeometryNode) {
            cout << *pGeometryNode << endl;
            weak_ptr<SceneObjectGeometry> pGeometry = scene->GetGeometry(pGeometryNode->GetSceneObjectRef());
            auto pObj = pGeometry.lock();
            if (pObj)
                cout << *pObj << endl;
        }
    }

    cout << "Dump of Cameras" << endl;
    cout << "---------------------------" << endl;
    for (const auto& _it : scene->CameraNodes)
    {
        auto pCameraNode = _it.second.lock();
        if (pCameraNode) {
            cout << *pCameraNode << endl;
            weak_ptr<SceneObjectCamera> pCamera = scene->GetCamera(pCameraNode->GetSceneObjectRef());
            auto pObj = pCamera.lock();
            if (pObj)
                cout << *pObj << endl;
        }
    }

    cout << "Dump of Lights" << endl;
    cout << "---------------------------" << endl;
    for (const auto& _it : scene->LightNodes)
    {
        auto pLightNode = _it.second.lock();
        if (pLightNode) {
            cout << *pLightNode << endl;
            weak_ptr<SceneObjectLight> pLight = scene->GetLight(pLightNode->GetSceneObjectRef());
            auto pObj = pLight.lock();
            if (pObj)
                cout << *pObj << endl;
        }
    }


    cout << "Dump of Materials" << endl;
    cout << "---------------------------" << endl;
    for (const auto& _it : scene->Materials)
    {
        auto pMaterial = _it.second;
        if (pMaterial)
            cout << *pMaterial << endl;
    }

    cout << "Dump of Bone Nodes" << endl;
    cout << "---------------------------" << endl;
    for (const auto& _it : scene->BoneNodes)
    {
        auto pBone = _it.second.lock();
        if (pBone)
            cout << *pBone << endl;
    }
}

int main(int argc, char** argv) {
    int error = 0;

    BaseApplication app;
    AssetLoader assetLoader;
    SceneManager sceneManager;

    app.RegisterManagerModule(&assetLoader, &assetLoader);
    app.RegisterManagerModule(&sceneManager, &sceneManager);

    error = app.Initialize();

    if (argc >= 2) {
        sceneManager.LoadScene(argv[1]);
    } else {
        sceneManager.LoadScene("Scene/splash.ogex");
    }

    app.Finalize();

    return error;
}

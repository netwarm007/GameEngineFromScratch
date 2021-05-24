#include "Scene.hpp"

using namespace My;
using namespace std;

shared_ptr<SceneObjectCamera> Scene::GetCamera(const std::string& key) const {
    auto i = Cameras.find(key);
    if (i == Cameras.end()) {
        return nullptr;
    }

    return i->second;
}

shared_ptr<SceneObjectLight> Scene::GetLight(const std::string& key) const {
    auto i = Lights.find(key);
    if (i == Lights.end()) {
        return nullptr;
    }

    return i->second;
}

shared_ptr<SceneObjectGeometry> Scene::GetGeometry(
    const std::string& key) const {
    auto i = Geometries.find(key);
    if (i == Geometries.end()) {
        return nullptr;
    }

    return i->second;
}

shared_ptr<SceneObjectMaterial> Scene::GetMaterial(
    const std::string& key) const {
    auto i = Materials.find(key);
    if (i == Materials.end()) {
        return m_pDefaultMaterial;
    }

    return i->second;
}

shared_ptr<SceneObjectMaterial> Scene::GetFirstMaterial() const {
    return (Materials.empty() ? nullptr : Materials.cbegin()->second);
}

shared_ptr<SceneGeometryNode> Scene::GetFirstGeometryNode() const {
    return (GeometryNodes.empty() ? nullptr
                                  : GeometryNodes.cbegin()->second.lock());
}

shared_ptr<SceneLightNode> Scene::GetFirstLightNode() const {
    return (LightNodes.empty() ? nullptr : LightNodes.cbegin()->second.lock());
}

shared_ptr<SceneCameraNode> Scene::GetFirstCameraNode() const {
    return (CameraNodes.empty() ? nullptr
                                : CameraNodes.cbegin()->second.lock());
}

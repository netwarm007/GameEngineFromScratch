#include "SceneManager.hpp"

#include <iostream>

using namespace My;
using namespace std;

SceneManager::~SceneManager() = default;

int SceneManager::Initialize() {
    int result = 0;

    return result;
}

void SceneManager::Finalize() {}

void SceneManager::Tick() {}

int SceneManager::LoadScene(const char* scene_file_name) {
    if (!scene_file_name || !*scene_file_name) {
        cerr << "[SceneManager] Scene filename is empty" << endl;
        return -1;
    }

    string scene_text =
        m_Assets.SyncOpenAndReadTextFileToString(scene_file_name);

    if (scene_text.empty()) {
        cerr << "[SceneManager] Scene data is empty or unavailable: "
             << scene_file_name << endl;
        return -1;
    }

    auto scene = m_Parser.Parse(scene_text);
    if (!scene) {
        cerr << "[SceneManager] Scene parse failed: " << scene_file_name << endl;
        return -1;
    }

    m_pScene = std::move(scene);
    ++m_nSceneRevision;
    return 0;
}

void SceneManager::ResetScene() { m_nSceneRevision++; }

const std::shared_ptr<Scene> SceneManager::GetSceneForRendering() const {
    // TODO: we should perform CPU scene crop at here
    return m_pScene;
}

const std::shared_ptr<Scene> SceneManager::GetSceneForPhysicalSimulation()
    const {
    // TODO: we should perform CPU scene crop at here
    return m_pScene;
}

weak_ptr<BaseSceneNode> SceneManager::GetRootNode() const {
    return m_pScene->SceneGraph;
}

weak_ptr<SceneGeometryNode> SceneManager::GetSceneGeometryNode(
    const string& name) const {
    auto it = m_pScene->LUT_Name_GeometryNode.find(name);
    if (it != m_pScene->LUT_Name_GeometryNode.end()) {
        return it->second;
    }

    return weak_ptr<SceneGeometryNode>();
}

weak_ptr<SceneObjectGeometry> SceneManager::GetSceneGeometryObject(
    const string& key) const {
    return m_pScene->Geometries.find(key)->second;
}

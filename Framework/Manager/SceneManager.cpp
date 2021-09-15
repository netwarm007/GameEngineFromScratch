#include "SceneManager.hpp"

#include "AssetLoader.hpp"
#include "OGEX.hpp"

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
    // now we only has ogex scene parser, call it directly
    if (LoadOgexScene(scene_file_name)) {
        m_nSceneRevision++;
        return 0;
    }

    return -1;
}

void SceneManager::ResetScene() { m_nSceneRevision++; }

bool SceneManager::LoadOgexScene(const char* ogex_scene_file_name) {
    string ogex_text =
        g_pAssetLoader->SyncOpenAndReadTextFileToString(ogex_scene_file_name);

    if (ogex_text.empty()) {
        return false;
    }

    OgexParser ogex_parser;
    m_pScene = ogex_parser.Parse(ogex_text);

    return static_cast<bool>(m_pScene);
}

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

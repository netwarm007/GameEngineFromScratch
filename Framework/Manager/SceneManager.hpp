#pragma once
#include "IRuntimeModule.hpp"
#include "ISceneParser.hpp"
#include "geommath.hpp"

namespace My {
class SceneManager : _implements_ IRuntimeModule {
   public:
    ~SceneManager() override;

    int Initialize() override;
    void Finalize() override;

    void Tick() override;

    int LoadScene(const char* scene_file_name);

    uint64_t GetSceneRevision() const { return m_nSceneRevision; }

    const std::shared_ptr<Scene> GetSceneForRendering() const;
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const;

    void ResetScene();

    std::weak_ptr<BaseSceneNode> GetRootNode() const;
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(
        const std::string& name) const;
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(
        const std::string& key) const;

   protected:
    bool LoadOgexScene(const char* ogex_scene_file_name);

   protected:
    std::shared_ptr<Scene> m_pScene;
    uint64_t m_nSceneRevision = 0;
};

extern SceneManager* g_pSceneManager;
}  // namespace My

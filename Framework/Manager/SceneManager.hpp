#pragma once
#include "IRuntimeModule.hpp"
#include "ISceneManager.hpp"
#include "ISceneParser.hpp"
#include "geommath.hpp"

namespace My {
class SceneManager : _implements_ ISceneManager, _implements_ IRuntimeModule {
   public:
    SceneManager() {}
    ~SceneManager() override;

    int Initialize() override;
    void Finalize() override;

    void Tick() override;

    int LoadScene(const char* scene_file_name) override;

    uint64_t GetSceneRevision() const override { return m_nSceneRevision; }

    const std::shared_ptr<Scene> GetSceneForRendering() const override;
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const override;

    void ResetScene() override;

    std::weak_ptr<BaseSceneNode> GetRootNode() const override;
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(
        const std::string& name) const override;
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(
        const std::string& key) const override;

   protected:
    bool LoadOgexScene(const char* ogex_scene_file_name);

   protected:
    std::shared_ptr<Scene> m_pScene;
    uint64_t m_nSceneRevision = 0;
};
}  // namespace My

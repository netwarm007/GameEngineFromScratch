#pragma once
#include <memory>
#include "IRuntimeModule.hpp"
#include "Scene.hpp"

namespace My {
_Interface_ ISceneManager : _inherits_ IRuntimeModule {
   public:
    ISceneManager() = default;
    virtual ~ISceneManager() = default;

    virtual int LoadScene(const char* scene_file_name) = 0;

    virtual uint64_t GetSceneRevision() const = 0;

    virtual const std::shared_ptr<Scene> GetSceneForRendering() const = 0;
    virtual const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const = 0;

    virtual void ResetScene() = 0;

    virtual std::weak_ptr<BaseSceneNode> GetRootNode() const = 0;
    virtual std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(
        const std::string& name) const = 0;
    virtual std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(
        const std::string& key) const = 0;
};
}  // namespace My
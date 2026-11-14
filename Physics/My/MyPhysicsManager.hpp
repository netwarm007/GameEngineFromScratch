#pragma once
#include <map>

#include "PhysicsManager.hpp"
#include "RigidBody.hpp"

namespace My {
class Scene;
class ISceneManager;

class MyPhysicsManager : public PhysicsManager {
   public:
    using float_precision = float;

    explicit MyPhysicsManager(ISceneManager& scenes) : m_Scenes(scenes) {}
    ~MyPhysicsManager() override = default;

    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void CreateRigidBody(const std::shared_ptr<SceneGeometryNode>& node,
                         const SceneObjectGeometry& geometry) final;
    void DeleteRigidBody(SceneGeometryNode& node) final;

    int CreateRigidBodies() final;
    void ClearRigidBodies() final;

    bool UpdateRigidBodyTransform(SceneGeometryNode& node) final;

    bool ApplyCentralForce(SceneGeometryNode& node, Vector3f force) final;

    void IterateConvexHull();

   private:
    ISceneManager& m_Scenes;
    uint64_t m_nSceneRevision{0};
    struct BodyBinding {
        std::shared_ptr<SceneGeometryNode> node;
        std::unique_ptr<RigidBody<float_precision>> body;
    };
    std::map<SceneGeometryNode*, BodyBinding> m_RigidBodies;
    std::shared_ptr<Scene> m_pSimulationScene;
};
}  // namespace My
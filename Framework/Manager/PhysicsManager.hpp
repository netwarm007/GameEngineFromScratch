#pragma once
#include "Geometry.hpp"
#include "IPhysicsManager.hpp"

namespace My {
_Interface_ PhysicsManager : _implements_ IPhysicsManager {
   public:
    PhysicsManager() = default;
    ~PhysicsManager() override = default;

    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}

    void CreateRigidBody(const std::shared_ptr<SceneGeometryNode>& node,
                         const SceneObjectGeometry& geometry) override {}
    void DeleteRigidBody(SceneGeometryNode & node) override {}

    int CreateRigidBodies() override { return 0; }
    void ClearRigidBodies() override {}

    bool UpdateRigidBodyTransform(SceneGeometryNode & node) override { return false; }

    bool ApplyCentralForce(SceneGeometryNode& node, Vector3f force) override { return false; }
};
}  // namespace My
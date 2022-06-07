#pragma once
#include "Geometry.hpp"
#include "IPhysicsManager.hpp"
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ PhysicsManager : _implements_ IPhysicsManager,
                              _implements_ IRuntimeModule {
   public:
    PhysicsManager() = default;
    ~PhysicsManager() override = default;

    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}

    void CreateRigidBody(SceneGeometryNode & node,
                         const SceneObjectGeometry& geometry) override {}
    void DeleteRigidBody(SceneGeometryNode & node) override {}

    int CreateRigidBodies() override { return 0; }
    void ClearRigidBodies() override {}

    Matrix4X4f GetRigidBodyTransform(void* rigidBody) override { return Matrix4X4f(); }
    void UpdateRigidBodyTransform(SceneGeometryNode & node) override {}

    void ApplyCentralForce(void* rigidBody, Vector3f force) override {}
};
}  // namespace My
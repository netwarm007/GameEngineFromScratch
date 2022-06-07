#pragma once
#include "PhysicsManager.hpp"

namespace My {
class MyPhysicsManager : public PhysicsManager {
   public:
    MyPhysicsManager() = default;
    ~MyPhysicsManager() override = default;

    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void CreateRigidBody(SceneGeometryNode& node,
                         const SceneObjectGeometry& geometry) final;
    void DeleteRigidBody(SceneGeometryNode& node) final;

    int CreateRigidBodies() final;
    void ClearRigidBodies() final;

    Matrix4X4f GetRigidBodyTransform(void* rigidBody) final;
    void UpdateRigidBodyTransform(SceneGeometryNode& node) final;

    void ApplyCentralForce(void* rigidBody, Vector3f force) final;

    void IterateConvexHull();

   private:
    uint64_t m_nSceneRevision{0};
};
}  // namespace My
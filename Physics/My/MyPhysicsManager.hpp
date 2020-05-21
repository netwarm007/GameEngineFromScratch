#pragma once
#include "Geometry.hpp"
#include "IPhysicsManager.hpp"

namespace My {
class MyPhysicsManager : public IPhysicsManager {
   public:
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void CreateRigidBody(SceneGeometryNode& node,
                         const SceneObjectGeometry& geometry) override;
    void DeleteRigidBody(SceneGeometryNode& node) override;

    int CreateRigidBodies() override;
    void ClearRigidBodies() override;

    Matrix4X4f GetRigidBodyTransform(void* rigidBody) override;
    void UpdateRigidBodyTransform(SceneGeometryNode& node) override;

    void ApplyCentralForce(void* rigidBody, Vector3f force) override;

    static void IterateConvexHull();

#ifdef DEBUG
    void DrawDebugInfo() override;
#endif

   protected:
#ifdef DEBUG
    static void DrawAabb(const Geometry& geometry, const Matrix4X4f& trans,
                         const Vector3f& centerOfMass);
    static void DrawShape(const Geometry& geometry, const Matrix4X4f& trans,
                          const Vector3f& centerOfMass);
#endif
   private:
    uint64_t m_nSceneRevision{0};
};
}  // namespace My
#pragma once
#include "IPhysicsManager.hpp"
#define BT_USE_DOUBLE_PRECISION 1
#include <btBulletDynamicsCommon.h>

namespace My {
class BulletPhysicsManager : public IPhysicsManager {
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

   protected:
    uint64_t m_nSceneRevision{0};
    btBroadphaseInterface* m_btBroadphase;
    btDefaultCollisionConfiguration* m_btCollisionConfiguration;
    btCollisionDispatcher* m_btDispatcher;
    btSequentialImpulseConstraintSolver* m_btSolver;
    btDiscreteDynamicsWorld* m_btDynamicsWorld;

    std::vector<btCollisionShape*> m_btCollisionShapes;
};
}  // namespace My

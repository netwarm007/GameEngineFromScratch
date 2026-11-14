#pragma once
#include <map>

#include "PhysicsManager.hpp"
#define BT_USE_DOUBLE_PRECISION 1
#include <btBulletDynamicsCommon.h>

namespace My {
class Scene;

class BulletPhysicsManager : public PhysicsManager {
   public:
    BulletPhysicsManager() = default;
    ~BulletPhysicsManager() override = default;
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void CreateRigidBody(const std::shared_ptr<SceneGeometryNode>& node,
                         const SceneObjectGeometry& geometry) override;
    void DeleteRigidBody(SceneGeometryNode& node) override;

    int CreateRigidBodies() override;
    void ClearRigidBodies() override;

    bool UpdateRigidBodyTransform(SceneGeometryNode& node) override;

    bool ApplyCentralForce(SceneGeometryNode& node, Vector3f force) override;

   protected:
    Matrix4X4f GetRigidBodyTransform(const btRigidBody& rigidBody) const;
    void PublishTransforms();

    uint64_t m_nSceneRevision{0};
    btBroadphaseInterface* m_btBroadphase;
    btDefaultCollisionConfiguration* m_btCollisionConfiguration;
    btCollisionDispatcher* m_btDispatcher;
    btSequentialImpulseConstraintSolver* m_btSolver;
    btDiscreteDynamicsWorld* m_btDynamicsWorld;

    std::vector<btCollisionShape*> m_btCollisionShapes;
    struct BodyBinding {
        std::shared_ptr<SceneGeometryNode> node;
        btRigidBody* body;
    };
    std::map<SceneGeometryNode*, BodyBinding> m_RigidBodies;
    std::shared_ptr<Scene> m_pSimulationScene;
};
}  // namespace My

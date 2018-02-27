#pragma once
#include "IPhysicsManager.hpp"
#define BT_USE_DOUBLE_PRECISION 1
#include <btBulletDynamicsCommon.h>

namespace My {
    class BulletPhysicsManager : public IPhysicsManager
    {
    public:
        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

        virtual void CreateRigidBody(SceneGeometryNode& node, const SceneObjectGeometry& geometry);
        virtual void DeleteRigidBody(SceneGeometryNode& node);

        virtual int CreateRigidBodies();
        virtual void ClearRigidBodies();

        Matrix4X4f GetRigidBodyTransform(void* rigidBody);
        void UpdateRigidBodyTransform(SceneGeometryNode& node);

        void ApplyCentralForce(void* rigidBody, Vector3f force);

    protected:
        btBroadphaseInterface*                  m_btBroadphase;
        btDefaultCollisionConfiguration*        m_btCollisionConfiguration;
        btCollisionDispatcher*                  m_btDispatcher;
        btSequentialImpulseConstraintSolver*    m_btSolver;
        btDiscreteDynamicsWorld*                m_btDynamicsWorld;

        std::vector<btCollisionShape*>          m_btCollisionShapes;
    };
}


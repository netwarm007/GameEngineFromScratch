#pragma once
#include <vector>
#define BT_USE_DOUBLE_PRECISION 1
#include <btBulletDynamicsCommon.h>
#include "IRuntimeModule.hpp"
#include "SceneManager.hpp"

namespace My {
    class PhysicsManager : implements IRuntimeModule
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

    protected:

    protected:
        btBroadphaseInterface*                  m_btBroadphase;
        btDefaultCollisionConfiguration*        m_btCollisionConfiguration;
        btCollisionDispatcher*                  m_btDispatcher;
        btSequentialImpulseConstraintSolver*    m_btSolver;
        btDiscreteDynamicsWorld*                m_btDynamicsWorld;

        std::vector<btCollisionShape*>          m_btCollisionShapes;
    };

    extern PhysicsManager* g_pPhysicsManager;
}


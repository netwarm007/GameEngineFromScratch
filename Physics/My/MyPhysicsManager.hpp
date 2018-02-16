#pragma once
#include "IPhysicsManager.hpp"
#include "Geometry.hpp"

namespace My {
    class MyPhysicsManager : public IPhysicsManager
    {
    public:
        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

        virtual void CreateRigidBody(SceneGeometryNode& node, const SceneObjectGeometry& geometry);
        virtual void DeleteRigidBody(SceneGeometryNode& node);

        virtual int CreateRigidBodies();
        virtual void ClearRigidBodies();

        virtual Matrix4X4f GetRigidBodyTransform(void* rigidBody);

        virtual void ApplyCentralForce(void* rigidBody, Vector3f force);

#ifdef DEBUG
	    void DrawDebugInfo();
#endif

    protected:
        std::vector<Geometry*>          m_CollisionShapes;
    };
}
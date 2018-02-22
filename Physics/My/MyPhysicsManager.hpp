#pragma once
#include "IPhysicsManager.hpp"
#include "Geometry.hpp"

namespace My {
    class MyPhysicsManager : public IPhysicsManager
    {
    public:
        int Initialize();
        void Finalize();
        void Tick();

        void CreateRigidBody(SceneGeometryNode& node, const SceneObjectGeometry& geometry);
        void DeleteRigidBody(SceneGeometryNode& node);

        int CreateRigidBodies();
        void ClearRigidBodies();

        Matrix4X4f GetRigidBodyTransform(void* rigidBody);
        void UpdateRigidBodyTransform(SceneGeometryNode& node);

        void ApplyCentralForce(void* rigidBody, Vector3f force);

#ifdef DEBUG
	    void DrawDebugInfo();
#endif

    protected:
#ifdef DEBUG
        void DrawAabb(const Geometry& geometry, const Matrix4X4f& trans, const Vector3f& centerOfMass);
#endif
    };
}
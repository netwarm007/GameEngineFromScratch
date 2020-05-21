#pragma once
#include <vector>

#include "IRuntimeModule.hpp"
#include "SceneManager.hpp"

namespace My {
_Interface_ IPhysicsManager : _inherits_ IRuntimeModule {
   public:
    int Initialize() override = 0;
    void Finalize() override = 0;
    void Tick() override = 0;

    virtual void CreateRigidBody(SceneGeometryNode & node,
                                 const SceneObjectGeometry& geometry) = 0;
    virtual void DeleteRigidBody(SceneGeometryNode & node) = 0;

    virtual int CreateRigidBodies() = 0;
    virtual void ClearRigidBodies() = 0;

    virtual Matrix4X4f GetRigidBodyTransform(void* rigidBody) = 0;
    virtual void UpdateRigidBodyTransform(SceneGeometryNode & node) = 0;

    virtual void ApplyCentralForce(void* rigidBody, Vector3f force) = 0;
};

extern IPhysicsManager* g_pPhysicsManager;
}  // namespace My

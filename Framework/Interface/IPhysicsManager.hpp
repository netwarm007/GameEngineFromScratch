#pragma once
#include <vector>

#include "IRuntimeModule.hpp"
#include "SceneGeometryNode.hpp"
#include "SceneObjectGeometry.hpp"

namespace My {
_Interface_ IPhysicsManager : _inherits_ IRuntimeModule {
   public:
    IPhysicsManager() = default;
    virtual ~IPhysicsManager() = default;

    virtual void CreateRigidBody(SceneGeometryNode & node,
                                 const SceneObjectGeometry& geometry) = 0;
    virtual void DeleteRigidBody(SceneGeometryNode & node) = 0;

    virtual int CreateRigidBodies() = 0;
    virtual void ClearRigidBodies() = 0;

    virtual Matrix4X4f GetRigidBodyTransform(void* rigidBody) = 0;
    virtual void UpdateRigidBodyTransform(SceneGeometryNode & node) = 0;

    virtual void ApplyCentralForce(void* rigidBody, Vector3f force) = 0;
};
}  // namespace My

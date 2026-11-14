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

    virtual void CreateRigidBody(const std::shared_ptr<SceneGeometryNode>& node,
                                 const SceneObjectGeometry& geometry) = 0;
    virtual void DeleteRigidBody(SceneGeometryNode & node) = 0;

    virtual int CreateRigidBodies() = 0;
    virtual void ClearRigidBodies() = 0;

    virtual bool UpdateRigidBodyTransform(SceneGeometryNode & node) = 0;

    // A node without a body is a normal scene condition, reported as false.
    virtual bool ApplyCentralForce(SceneGeometryNode& node, Vector3f force) = 0;
};
}  // namespace My

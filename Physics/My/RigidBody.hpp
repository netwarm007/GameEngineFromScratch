#pragma once
#include <memory>
#include "Geometry.hpp"
#include "MotionState.hpp"

namespace My {
    class RigidBody {
    public:
        RigidBody(std::shared_ptr<Geometry> collisionShape, std::shared_ptr<MotionState> state) 
            : m_pCollisionShape(collisionShape), m_pMotionState(state) {}
        std::shared_ptr<MotionState> GetMotionState() { return m_pMotionState; }
        std::shared_ptr<Geometry>    GetCollisionShape() { return m_pCollisionShape; }

    private:
        std::shared_ptr<Geometry>       m_pCollisionShape;
        std::shared_ptr<MotionState>    m_pMotionState;
    };
}
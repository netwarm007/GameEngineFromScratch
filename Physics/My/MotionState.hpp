#pragma once
#include "geommath.hpp"

namespace My {
    class MotionState
    {
    public:
        MotionState(Matrix4X4f transition) : m_Transition(transition) {}
        void SetTransition(const Matrix4X4f& transition) { m_Transition = transition; }
        const Matrix4X4f& GetTransition() const { return m_Transition; }

    private:
        Matrix4X4f m_Transition;
    };
}
#pragma once
#include "geommath.hpp"

namespace My {
class MotionState {
   public:
    explicit MotionState(const Matrix4X4f& transition)
        : m_Transition(transition), m_CenterOfMassOffset(0) {}
    MotionState(const Matrix4X4f& transition, const Vector3f& centroid)
        : m_Transition(transition), m_CenterOfMassOffset(centroid) {}
    void SetTransition(const Matrix4X4f& transition) {
        m_Transition = transition;
    }
    void SetCenterOfMass(const Vector3f& centroid) {
        m_CenterOfMassOffset = centroid;
    }
    [[nodiscard]] Matrix4X4f GetTransition() const { return m_Transition; }
    [[nodiscard]] Vector3f GetCenterOfMassOffset() const {
        return m_CenterOfMassOffset;
    }

   private:
    Matrix4X4f m_Transition;
    Vector3f m_CenterOfMassOffset;
};
}  // namespace My
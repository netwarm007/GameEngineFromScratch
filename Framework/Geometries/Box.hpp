#pragma once
#include "Geometry.hpp"

namespace My {
class Box : public Geometry {
   public:
    Box() = delete;
    explicit Box(Vector3f halfExtents)
        : Geometry(GeometryType::kBox), m_vHalfExtents(halfExtents) {}

    void GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                 Vector3f& aabbMax) const final;

    [[nodiscard]] Vector3f GetDimension() const {
        return m_vHalfExtents * 2.0f;
    }
    [[nodiscard]] Vector3f GetDimensionWithMargin() const {
        return m_vHalfExtents * 2.0f + m_fMargin;
    }
    [[nodiscard]] Vector3f GetHalfExtents() const { return m_vHalfExtents; }
    [[nodiscard]] Vector3f GetHalfExtentsWithMargin() const {
        return m_vHalfExtents + m_fMargin;
    }

   protected:
    Vector3f m_vHalfExtents;
};
}  // namespace My
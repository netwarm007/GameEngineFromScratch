#pragma once
#include "Geometry.hpp"

namespace My {
template <typename T>
class Box : public Geometry<T> {
   public:
    Box()
        : Geometry<T>(GeometryType::kBox),
          m_vHalfExtents({0.5, 0.5, 0.5}),
          m_vCenter({0, 0, 0}) {}
    Box(Vector3<T> halfExtents) : Box() { m_vHalfExtents = halfExtents; }
    Box(Vector3<T> halfExtents, Point<T> center) : Box() { m_vHalfExtents = halfExtents; m_vCenter = center; }

    bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const final {
        if (isNearZero(m_vHalfExtents)) return false;

        TransformAabb(m_vHalfExtents, Geometry<T>::m_fMargin, trans, aabb);
        return true;
    }

    [[nodiscard]] Vector3<T> GetDimension() const {
        return m_vHalfExtents * 2.0;
    }
    [[nodiscard]] Vector3<T> GetDimensionWithMargin() const {
        return m_vHalfExtents * 2.0 + Geometry<T>::m_fMargin;
    }
    [[nodiscard]] Vector3<T> GetHalfExtents() const { return m_vHalfExtents; }
    [[nodiscard]] Vector3<T> GetHalfExtentsWithMargin() const {
        return m_vHalfExtents + Geometry<T>::m_fMargin;
    }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        return false;
    }

   protected:
    Point<T> m_vCenter;
    Vector3<T> m_vHalfExtents;
};
}  // namespace My
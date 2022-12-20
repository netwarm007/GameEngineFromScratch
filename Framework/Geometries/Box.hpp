#pragma once
#include "Geometry.hpp"
#include "Intersectable.hpp"

namespace My {
template <typename T>
class Box : public Geometry, _implements_ Intersectable<T> {
   public:
    Box()
        : Geometry(GeometryType::kBox),
          m_vHalfExtents({0.5, 0.5, 0.5}),
          m_vCenter({0, 0, 0}) {}
    Box(Vector3<T> halfExtents) : Box() { m_vHalfExtents = halfExtents; }
    Box(Vector3<T> halfExtents, Point<T> center) : Box() { m_vHalfExtents = halfExtents; m_vCenter = center; }

    void GetAabb(const Matrix4X4<T>& trans, Vector3<T>& aabbMin,
                 Vector3<T>& aabbMax) const final {
        TransformAabb(m_vHalfExtents, m_fMargin, trans, aabbMin, aabbMax);
    }

    [[nodiscard]] Vector3<T> GetDimension() const {
        return m_vHalfExtents * 2.0;
    }
    [[nodiscard]] Vector3<T> GetDimensionWithMargin() const {
        return m_vHalfExtents * 2.0 + m_fMargin;
    }
    [[nodiscard]] Vector3<T> GetHalfExtents() const { return m_vHalfExtents; }
    [[nodiscard]] Vector3<T> GetHalfExtentsWithMargin() const {
        return m_vHalfExtents + m_fMargin;
    }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin) const override {
        bool result = false;

        return result;
    }

   protected:
    Point<T> m_vCenter;
    Vector3<T> m_vHalfExtents;
};
}  // namespace My
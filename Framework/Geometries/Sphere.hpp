#pragma once
#include "Geometry.hpp"
#include "Intersectable.hpp"

namespace My {
template <typename T>
class Sphere : public Geometry, _implements_ Intersectable<T> {
   public:
    Sphere() = delete;
    explicit Sphere(const T radius)
        : Geometry(GeometryType::kSphere), m_fRadius(radius) {}

    explicit Sphere(const T radius, const Point<T> center)
        : Geometry(GeometryType::kSphere),
          m_fRadius(radius),
          m_center(center) {}

    explicit Sphere(const T radius, const Point<T> center,
                    const std::shared_ptr<material> m)
        : Geometry(GeometryType::kSphere), m_fRadius(radius), m_center(center) {
        m_ptrMat = m;
    }

    void GetAabb(const Matrix4X4<T>& trans, Vector3<T>& aabbMin,
                 Vector3<T>& aabbMax) const final {
        Vector3f center;
        GetOrigin(center, trans);
        center += m_center;
        Vector3f extent({m_fMargin, m_fMargin, m_fMargin});
        aabbMin = center - extent;
        aabbMax = center + extent;
    }

    [[nodiscard]] T GetRadius() const { return m_fRadius; }
    [[nodiscard]] Vector3<T> GetCenter() const { return m_center; }

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        // Ray: R(t) = O + V dot t
        // Sphere: || X - C || = r
        // Intersect equation: at^2  + bt + c = 0; a = V dot V; b = 2V dot (O -
        // C); C = ||O - C||^2 - r^2 Intersect condition: b^2 - 4ac > 0
        const Vector3f& V = r.getDirection();
        const Vector3f& O = r.getOrigin();
        Vector3f tmp = O - m_center;
        T dist = Length(tmp);

        T half_b = V.Dot3(tmp);
        T c = dist * dist - m_fRadius * m_fRadius;
        T disc = half_b * half_b - c;

        if (disc < 0) return false;

        T sroot = sqrt(disc);

        T t = -half_b - sroot;
        if (t < tmin || tmax < t) {
            t = -half_b + sroot;
            if (t < tmin || tmax < t) {
                return false;
            }
        }

        // calculate normal
        auto p = r.pointAtParameter(t);
        auto normal = (p - m_center) / m_fRadius;
        bool front_face = DotProduct(V, normal) < 0;

        // set the hit result
        h.set(t, p, normal, front_face, m_ptrMat);

        return true;
    }

   protected:
    T m_fRadius;
    Point<T> m_center;
};
}  // namespace My
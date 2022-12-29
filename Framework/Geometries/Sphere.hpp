#pragma once
#include "Geometry.hpp"
#include "MaterialContainer.hpp"

namespace My {
template <class T, class MaterialPtr>
class Sphere : public Geometry<T>, _implements_ MaterialContainer<MaterialPtr> {
   public:
    Sphere() = delete;
    __device__ explicit Sphere(const T radius)
        : Geometry<T>(GeometryType::kSphere), m_fRadius(radius) {}

    __device__ explicit Sphere(const T radius, const Point<T> center)
        : Geometry<T>(GeometryType::kSphere),
          m_fRadius(radius),
          m_center(center) {}

    __device__ explicit Sphere(const T radius, const Point<T> center,
                    MaterialPtr m)
        : Geometry<T>(GeometryType::kSphere), m_fRadius(radius), m_center(center) {
        this->m_ptrMat = m;
    }

    __device__ bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const final {
        if (m_fRadius == 0.0) return false;

        Vector3<T> center;
        GetOrigin(center, trans);
        center += m_center;
        Vector3<T> extent({m_fRadius + this->m_fMargin, m_fRadius + this->m_fMargin, m_fRadius + this->m_fMargin});
        aabb = AaBb(center - extent, center + extent);

        return true;
    }

    [[nodiscard]] T GetRadius() const { return m_fRadius; }
    [[nodiscard]] Vector3<T> GetCenter() const { return m_center; }

    __device__ bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        // Ray: R(t) = O + V dot t
        // Sphere: || X - C || = r
        // Intersect equation: at^2  + bt + c = 0; a = V dot V; b = 2V dot (O -
        // C); C = ||O - C||^2 - r^2 Intersect condition: b^2 - 4ac > 0
        const Vector3<T>& V = r.getDirection();
        const Vector3<T>& O = r.getOrigin();
        Vector3<T> tmp = O - m_center;
        T dist = Length(tmp);

        T half_b = DotProduct(V, tmp);
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
        h.set(t, p, normal, front_face, &this->m_ptrMat);

        return true;
    }

   protected:
    T m_fRadius;
    Point<T> m_center;
};
}  // namespace My
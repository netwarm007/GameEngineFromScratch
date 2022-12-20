#pragma once
#include "Geometry.hpp"
#include "Intersectable.hpp"

namespace My {
template <typename T>
class Sphere : public Geometry, _implements_ Intersectable<T> {
   public:
    Sphere() = delete;
    explicit Sphere(const T radius)
        : Geometry(GeometryType::kSphere), m_fRadius(radius){};

    explicit Sphere(const T radius, const Vector3<T> center)
        : Geometry(GeometryType::kSphere),
          m_fRadius(radius),
          m_center(center){};

    void GetAabb(const Matrix4X4<T>& trans, Vector3<T>& aabbMin,
                 Vector3<T>& aabbMax) const final {
        Vector3f center;
        GetOrigin(center, trans);
        center += m_center;
        Vector3f extent({m_fMargin, m_fMargin, m_fMargin});
        aabbMin = center - extent;
        aabbMax = center + extent;
    }

    [[nodiscard]] T GetRadius() const { return m_fRadius; };
    [[nodiscard]] Vector3<T> GetCenter() const { return m_center; };

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin) const override {
        bool result = false;

        // Ray: R(t) = O + V dot t
        // Sphere: || X - C || = r
        // Intersect equation: at^2  + bt + c = 0; a = V dot V; b = 2V dot (O -
        // C); C = ||O - C||^2 - r^2 Intersect condition: b^2 - 4ac > 0
        Vector3f V = r.getDirection();
        Vector3f O = r.getOrigin();
        Vector3f tmp = O - m_center;
        T dist = Length(tmp);

        T b = 2 * V.Dot3(tmp);
        T c = dist * dist - m_fRadius * m_fRadius;
        T disc = b * b - 4 * c;

        T t = std::numeric_limits<T>::infinity();

        if (disc > 0) {
            T sroot = sqrt(disc);

            T t1 = (-b - sroot) * 0.5;
            T t2 = (-b + sroot) * 0.5;

            if (t1 >= tmin)
                t = t1;
            else if (t2 >= tmin)
                t = t2;

            if (t < h.getT()) {
                h.set(t, m_color);
            }

            result = true;

        } else
            result = false;

        return result;
    }

   protected:
    T m_fRadius;
    Point<T> m_center;
};
}  // namespace My
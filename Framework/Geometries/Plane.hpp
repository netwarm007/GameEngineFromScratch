#pragma once
#include "Geometry.hpp"

namespace My {
template <class T>
class Plane : public Geometry<T> {
   public:
    Plane() = delete;
    Plane(Vector3<T> normal, T intercept)
        : Geometry<T>(GeometryType::kPlane),
          m_vNormal(normal),
          m_fIntercept(intercept){};

    bool GetAabb(const Matrix4X4<T>& trans, AaBb<T, 3>& aabb) const final {
        (void)trans;
        T minf = std::numeric_limits<T>::lowest();
        T maxf = std::numeric_limits<T>::max();
        Vector3<T> aabbMin = {minf, minf, minf};
        Vector3<T> aabbMax = {maxf, maxf, maxf};
        aabb = AaBb<T, 3>(aabbMin, aabbMax);

        return true;
    }

    [[nodiscard]] Vector3<T> GetNormal() const { return m_vNormal; };
    [[nodiscard]] T GetIntercept() const { return m_fIntercept; };

    bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax) const override {
        return true;
    }

   protected:
    Vector3<T> m_vNormal;
    T m_fIntercept;
};
}  // namespace My
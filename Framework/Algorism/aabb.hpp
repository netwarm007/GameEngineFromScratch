#pragma once
#include "geommath.hpp"
#include <cmath>
#include <utility>

namespace My {
template <class T, Dimension auto N>
class AaBb {
   public:
    AaBb() {}
    AaBb(const Vector<T, N>& a, const Vector<T, N>& b) {
        minimum = a;
        maximum = b;
    }

    Vector<T, N> min() const { return minimum; }
    Vector<T, N> max() const { return maximum; }


   private:
    Vector<T, N> minimum;
    Vector<T, N> maximum;
};

template <class T>
inline void TransformAabb(const Vector3f& halfExtents, float margin,
                          const Matrix4X4f& trans, AaBb<T, 3>& aabb) {
    Vector3f halfExtentsWithMargin = halfExtents + Vector3f(margin);
    Vector3f center;
    Vector3f extent;
    Matrix3X3f basis;
    GetOrigin(center, trans);
    Shrink(basis, trans);
    Absolute(basis, basis);
    DotProduct3(extent, halfExtentsWithMargin, basis);
    aabb = AaBb<T, 3>(center - extent, center + extent);
};

template <class T>
inline auto SurroundingBox(AaBb<T, 3> box0, AaBb<T, 3> box1) {
    Vector3<T> small({
        std::fmin(box0.min()[0], box1.min()[0]),
        std::fmin(box0.min()[1], box1.min()[1]),
        std::fmin(box0.min()[2], box1.min()[2])
    });

    Vector3<T> big({
        std::fmax(box0.max()[0], box1.max()[0]),
        std::fmax(box0.max()[1], box1.max()[1]),
        std::fmax(box0.max()[2], box1.max()[2])
    });

    return AaBb<T, 3>(small, big);
};
}  // namespace My
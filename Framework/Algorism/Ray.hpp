#pragma once
#include <iostream>

#include "geommath.hpp"

namespace My {
template <typename T>
class Ray {
   public:
    // CONSTRUCTOR & DESTRUCTOR
    __device__ Ray() {}
    __device__ Ray(const Vector3<T> &orig, const Vector3<T> &dir) {
        direction = dir;
        Normalize(direction);
        origin = orig;
    }

    __device__ Ray(const Ray &r) { *this = r; }

    // ACCESSORS
    __device__ const auto getOrigin() const { return origin; }
    __device__ const auto getDirection() const { return direction; }
    __device__ Point<T> pointAtParameter(T t) const { return origin + direction * t; }

   private:
    // REPRESENTATION
    Vector3<T> direction;
    Point<T> origin;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Ray<T> &r) {
    os << "Ray <" << r.getOrigin() << ", " << r.getDirection() << ">";
    return os;
}
}  // namespace My
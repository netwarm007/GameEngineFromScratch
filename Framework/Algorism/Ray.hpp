#pragma once
#include <iostream>

#include "geommath.hpp"

namespace My {
template <typename T>
class Ray {
   public:
    // CONSTRUCTOR & DESTRUCTOR
    Ray() {}
    Ray(const Vector3<T> &dir, const Vector3<T> &orig) {
        direction = dir;
        Normalize(direction);
        origin = orig;
    }
    Ray(const Ray &r) { *this = r; }

    // ACCESSORS
    const Vector3<T> &getOrigin() const { return origin; }
    const Vector3<T> &getDirection() const { return direction; }
    Vector3<T> pointAtParameter(T t) const { return origin + direction * t; }

   private:
    // REPRESENTATION
    Vector3<T> direction;
    Vector3<T> origin;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Ray<T> &r) {
    os << "Ray <" << r.getOrigin() << ", " << r.getDirection() << ">";
    return os;
}
}  // namespace My
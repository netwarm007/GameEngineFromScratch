#pragma once
#include <iostream>

#include "geommath.hpp"

namespace My {
template <typename T>
class Ray {
   public:
    // CONSTRUCTOR & DESTRUCTOR
    Ray() {}
    Ray(const Vector3<T> &orig, const Vector3<T> &dir) {
        direction = dir;
        Normalize(direction);
        origin = orig;
    }
    Ray(const Ray &r) { *this = r; }

    // ACCESSORS
    const auto getOrigin() const { return origin; }
    const auto getDirection() const { return direction; }
    Point<T> pointAtParameter(T t) const { return origin + direction * t; }

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
#pragma once
#include <iostream>

#include "geommath.hpp"

namespace My {
class Ray {
   public:
    // CONSTRUCTOR & DESTRUCTOR
    Ray() {}
    Ray(const Vector3f &dir, const Vector3f &orig) {
        direction = dir;
        Normalize(direction);
        origin = orig;
    }
    Ray(const Ray &r) { *this = r; }

    // ACCESSORS
    const Vector3f &getOrigin() const { return origin; }
    const Vector3f &getDirection() const { return direction; }
    Vector3f pointAtParameter(float t) const { return origin + direction * t; }

   private:
    // REPRESENTATION
    Vector3f direction;
    Vector3f origin;
};

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << "Ray <" << r.getOrigin() << ", " << r.getDirection() << ">";
    return os;
}
}  // namespace My
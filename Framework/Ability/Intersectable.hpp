#pragma once
#include "Ability.hpp"
#include "Ray.hpp"
#include "Hit.hpp"

namespace My {
template <typename T>
Ability Intersectable {
   public:
    virtual ~Intersectable() = default;
    using ParamType = const T;
    virtual bool Intersect(const Ray &r, Hit &h, float tmin) const  = 0;
};
}  // namespace My

#pragma once
#include "Ability.hpp"
#include "Hit.hpp"
#include "Ray.hpp"

namespace My {
template <class T>
Ability Intersectable {
   public:
    __device__ virtual ~Intersectable() {}
    using ParamType = T;

    __device__ virtual bool Intersect(const Ray<T>& r, Hit<T>& h, T tmin, T tmax)
        const = 0;
};
}  // namespace My

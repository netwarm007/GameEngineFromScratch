#pragma once
#include <vector>
#include "portable.hpp"

namespace My {
    ENUM(CurveType) {
        kLinear = "LINE"_i32,
        kBezier = "BEZI"_i32
    };
    
    template <typename T>
    struct Curve 
    {
        Curve() = default;
        virtual ~Curve() = default;
        virtual T Reverse(T p) const = 0; 
        virtual T Interpolate(T t) const = 0;
        virtual CurveType GetCurveType() const = 0;
    };
}

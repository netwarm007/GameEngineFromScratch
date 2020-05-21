#pragma once
#include "Ability.hpp"

namespace My {
template <typename T>
Ability Animatable {
   public:
    virtual ~Animatable() = default;
    using ParamType = const T;
    virtual void Update(ParamType param) = 0;
};
}  // namespace My

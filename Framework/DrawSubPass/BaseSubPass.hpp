#pragma once
#include "IDrawSubPass.hpp"

namespace My {
class BaseSubPass : _implements_ IDrawSubPass {
   public:
    void BeginSubPass() override{};
    void EndSubPass() override{};
};
}  // namespace My
#pragma once

#include "Interface.hpp"

namespace My {
class Scene;

_Interface_ ISceneResourceLoader {
   public:
    virtual ~ISceneResourceLoader() = default;
    virtual bool Load(Scene & scene) = 0;
};
}  // namespace My

#pragma once
#include "Interface.hpp"
#include "Scene.hpp"

namespace My {
_Interface_ ISceneParser {
   public:
    virtual std::unique_ptr<Scene> Parse(const std::string& buf) = 0;
};
}  // namespace My

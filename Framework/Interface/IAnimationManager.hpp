#pragma once
#include "Interface.hpp"
#include "SceneObjectAnimation.hpp"

namespace My {
_Interface_ IAnimationManager {
   public:
    IAnimationManager() = default;
    virtual ~IAnimationManager() = default;

    virtual void AddAnimationClip(
        const std::shared_ptr<SceneObjectAnimationClip>& clip) = 0;
    virtual void ClearAnimationClips() = 0;
};
}  // namespace My
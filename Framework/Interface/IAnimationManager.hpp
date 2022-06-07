#pragma once
#include "IRuntimeModule.hpp"
#include "SceneObjectAnimation.hpp"

namespace My {
_Interface_ IAnimationManager : _inherits_ IRuntimeModule {
   public:
    IAnimationManager() = default;
    virtual ~IAnimationManager() = default;

    virtual void AddAnimationClip(
        const std::shared_ptr<SceneObjectAnimationClip>& clip) = 0;
    virtual void ClearAnimationClips() = 0;
};
}  // namespace My
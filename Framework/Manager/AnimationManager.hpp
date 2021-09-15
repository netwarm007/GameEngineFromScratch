#pragma once
#include <chrono>
#include <list>

#include "IRuntimeModule.hpp"
#include "SceneObject.hpp"

namespace My {
class AnimationManager : public IRuntimeModule {
   public:
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void AddAnimationClip(
        const std::shared_ptr<SceneObjectAnimationClip>& clip);
    void ClearAnimationClips();

   private:
    uint64_t m_nSceneRevision{0};
    std::chrono::steady_clock::time_point m_TimeLineStartPoint;
    std::chrono::duration<float> m_TimeLineValue;
    std::list<std::shared_ptr<SceneObjectAnimationClip>> m_AnimationClips;
    bool m_bTimeLineStarted{false};
};

extern AnimationManager* g_pAnimationManager;
}  // namespace My
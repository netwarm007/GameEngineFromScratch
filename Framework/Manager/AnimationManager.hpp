#pragma once
#include <chrono>
#include <list>

#include "IAnimationManager.hpp"
#include "ISceneManager.hpp"
#include "SceneObject.hpp"

namespace My {
class AnimationManager : _implements_ IAnimationManager {
   public:
    explicit AnimationManager(ISceneManager& scenes) : m_Scenes(scenes) {}
    ~AnimationManager() override {}
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void AddAnimationClip(
        const std::shared_ptr<SceneObjectAnimationClip>& clip) override;
    void ClearAnimationClips() override;

   private:
    ISceneManager& m_Scenes;
    uint64_t m_nSceneRevision{0};
    std::chrono::steady_clock::time_point m_TimeLineStartPoint;
    std::chrono::duration<float> m_TimeLineValue;
    std::list<std::shared_ptr<SceneObjectAnimationClip>> m_AnimationClips;
    bool m_bTimeLineStarted{false};
};
}  // namespace My
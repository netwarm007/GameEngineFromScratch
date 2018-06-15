#pragma once
#include "IRuntimeModule.hpp"
#include "SceneObject.hpp"
#include <chrono>
#include <list>

namespace My {
    class AnimationManager : public IRuntimeModule
    {
    public:
        int Initialize();
        void Finalize();
        void Tick();

        void AddAnimationClip(std::shared_ptr<SceneObjectAnimationClip> clip);
        void ClearAnimationClips();

    private:
        std::chrono::steady_clock m_Clock;
        std::chrono::time_point<std::chrono::steady_clock> m_TimeLineStartPoint;
        std::chrono::duration<float> m_TimeLineValue;
        std::list<std::shared_ptr<SceneObjectAnimationClip>> m_AnimationClips;
        bool m_bTimeLineStarted = false;
    };

    extern AnimationManager* g_pAnimationManager;
}
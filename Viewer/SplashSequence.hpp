#pragma once

#include <memory>

#include "IRuntimeModule.hpp"
#include "SceneGeometryNode.hpp"

namespace My {
class AnimationManager;
class IPhysicsManager;
class ISceneManager;

class SplashSequence : public IRuntimeModule {
   public:
    SplashSequence(ISceneManager& scenes, IPhysicsManager& physics,
                   AnimationManager& animations)
        : m_Scenes(scenes), m_Physics(physics), m_Animations(animations) {}
    int Initialize() override { return 0; }
    void Finalize() override;
    void Tick() override;

   private:
    void RestoreNodes();

    ISceneManager& m_Scenes;
    IPhysicsManager& m_Physics;
    AnimationManager& m_Animations;
    uint64_t m_Revision{0};
    std::shared_ptr<SceneGeometryNode> m_Text;
    std::shared_ptr<SceneGeometryNode> m_Ball;
    bool m_TextPhysics{true};
    bool m_BallPhysics{true};
    bool m_BallVisible{true};
    float m_EndTime{0.0f};
    bool m_Waiting{false};
};
}  // namespace My

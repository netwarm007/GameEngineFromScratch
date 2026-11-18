#pragma once
#include "GameLogic.hpp"

namespace My {
class ISceneManager;
class IPhysicsManager;

class BilliardGameLogic : public GameLogic {
   public:
    BilliardGameLogic(ISceneManager& scenes, IPhysicsManager& physics)
        : m_Scenes(scenes), m_Physics(physics) {}

   private:
    ISceneManager& m_Scenes;
    IPhysicsManager& m_Physics;

    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void OnLeftKey() final;
    void OnAnalogStick(int id, float deltaX, float deltaY) final;
};
}  // namespace My
#pragma once
#include <string>

#include "IApplication.hpp"
#include "GameLogic.hpp"

namespace My {
class ISceneManager;

class ViewerLogic : public GameLogic {
   public:
    explicit ViewerLogic(ISceneManager& scenes) : m_Scenes(scenes) {}

   private:
    ISceneManager& m_Scenes;
    int m_SceneCycle = 0;
    int m_SceneCycleTick = 0;
    std::string m_SceneCycleReadyPath;
    // overrides
    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void OnLeftKeyDown() final;
    void OnRightKeyDown() final;
    void OnUpKeyDown() final;
    void OnDownKeyDown() final;

    void OnAnalogStick(int id, float deltaX, float deltaY) final;
};
}  // namespace My

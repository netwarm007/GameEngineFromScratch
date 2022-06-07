#pragma once
#include "IInputManager.hpp"

namespace My {
class InputManager : _implements_ IInputManager {
   public:
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    // keyboard handling
    void UpArrowKeyDown() override;
    void UpArrowKeyUp() override;
    void DownArrowKeyDown() override;
    void DownArrowKeyUp() override;
    void LeftArrowKeyDown() override;
    void LeftArrowKeyUp() override;
    void RightArrowKeyDown() override;
    void RightArrowKeyUp() override;

    void AsciiKeyDown(char keycode) override;
    void AsciiKeyUp(char keycode) override;

    // mouse handling
    void LeftMouseButtonDown() override;
    void LeftMouseButtonUp() override;
    void LeftMouseDrag(int deltaX, int deltaY) override;

    // mouse handling
    void RightMouseButtonDown() override;
    void RightMouseButtonUp() override;
    void RightMouseDrag(int deltaX, int deltaY) override;

   protected:
    bool m_bUpKeyPressed = false;
    bool m_bDownKeyPressed = false;
    bool m_bLeftKeyPressed = false;
    bool m_bRightKeyPressed = false;
};
}  // namespace My

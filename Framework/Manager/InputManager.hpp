#pragma once
#include "IRuntimeModule.hpp"

namespace My {
class InputManager : _implements_ IRuntimeModule {
   public:
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    // keyboard handling
    void UpArrowKeyDown();
    void UpArrowKeyUp();
    void DownArrowKeyDown();
    void DownArrowKeyUp();
    void LeftArrowKeyDown();
    void LeftArrowKeyUp();
    void RightArrowKeyDown();
    void RightArrowKeyUp();

    static void AsciiKeyDown(char keycode);
    static void AsciiKeyUp(char keycode);

    // mouse handling
    static void LeftMouseButtonDown();
    static void LeftMouseButtonUp();
    static void LeftMouseDrag(int deltaX, int deltaY);

    // mouse handling
    static void RightMouseButtonDown();
    static void RightMouseButtonUp();
    static void RightMouseDrag(int deltaX, int deltaY);

   protected:
    bool m_bUpKeyPressed = false;
    bool m_bDownKeyPressed = false;
    bool m_bLeftKeyPressed = false;
    bool m_bRightKeyPressed = false;
};

extern InputManager* g_pInputManager;
}  // namespace My

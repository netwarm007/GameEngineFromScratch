#pragma once
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IInputManager : _inherits_ IRuntimeModule {
   public:
    IInputManager() = default;
    virtual ~IInputManager() = default;
    // keyboard handling
    virtual void UpArrowKeyDown() = 0;
    virtual void UpArrowKeyUp() = 0;
    virtual void DownArrowKeyDown() = 0;
    virtual void DownArrowKeyUp() = 0;
    virtual void LeftArrowKeyDown() = 0;
    virtual void LeftArrowKeyUp() = 0;
    virtual void RightArrowKeyDown() = 0;
    virtual void RightArrowKeyUp() = 0;

    virtual void AsciiKeyDown(char keycode) = 0;
    virtual void AsciiKeyUp(char keycode) = 0;

    virtual void LeftMouseButtonDown() = 0;
    virtual void LeftMouseButtonUp() = 0;
    virtual void LeftMouseDrag(int deltaX, int deltaY) = 0;

    virtual void RightMouseButtonDown() = 0;
    virtual void RightMouseButtonUp() = 0;
    virtual void RightMouseDrag(int deltaX, int deltaY) = 0;
};
}  // namespace My
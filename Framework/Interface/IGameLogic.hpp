#pragma once
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IGameLogic : _inherits_ IRuntimeModule {
   public:
    IGameLogic() = default;
    virtual ~IGameLogic() = default;

    virtual void OnUpKeyDown() = 0;
    virtual void OnUpKeyUp() = 0;
    virtual void OnUpKey() = 0;

    virtual void OnDownKeyDown() = 0;
    virtual void OnDownKeyUp() = 0;
    virtual void OnDownKey() = 0;

    virtual void OnLeftKeyDown() = 0;
    virtual void OnLeftKeyUp() = 0;
    virtual void OnLeftKey() = 0;

    virtual void OnRightKeyDown() = 0;
    virtual void OnRightKeyUp() = 0;
    virtual void OnRightKey() = 0;

    virtual void OnButton1Down() = 0;
    virtual void OnButton1Up() = 0;

    virtual void OnAnalogStick(int id, float deltaX, float deltaY) = 0;
};
}  // namespace My
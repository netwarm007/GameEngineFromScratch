#include "IGameLogic.hpp"

namespace My {
class GameLogic : _implements_ IGameLogic {
    // overrides
    int Initialize() { return 0; }
    void Finalize() {}
    void Tick() {}

    virtual void OnUpKeyDown() {}
    virtual void OnUpKeyUp() {}
    virtual void OnUpKey() {}

    virtual void OnDownKeyDown() {}
    virtual void OnDownKeyUp() {}
    virtual void OnDownKey() {}

    virtual void OnLeftKeyDown() {}
    virtual void OnLeftKeyUp() {}
    virtual void OnLeftKey() {}

    virtual void OnRightKeyDown() {}
    virtual void OnRightKeyUp() {}
    virtual void OnRightKey() {}

    virtual void OnButton1Down() {}
    virtual void OnButton1Up() {}

    virtual void OnAnalogStick(int id, float deltaX, float deltaY) {}
};
}  // namespace My

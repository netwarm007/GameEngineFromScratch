#pragma once
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IGameLogic : _inherits_ IRuntimeModule {
   public:
    int Initialize() override = 0;
    void Finalize() override = 0;
    void Tick() override = 0;

    virtual void OnUpKeyDown(){};
    virtual void OnUpKeyUp(){};
    virtual void OnUpKey(){};

    virtual void OnDownKeyDown(){};
    virtual void OnDownKeyUp(){};
    virtual void OnDownKey(){};

    virtual void OnLeftKeyDown(){};
    virtual void OnLeftKeyUp(){};
    virtual void OnLeftKey(){};

    virtual void OnRightKeyDown(){};
    virtual void OnRightKeyUp(){};
    virtual void OnRightKey(){};

    virtual void OnButton1Down(){};
    virtual void OnButton1Up(){};

    virtual void OnAnalogStick(int id, float deltaX, float deltaY){};
};

extern IGameLogic* g_pGameLogic;
}  // namespace My
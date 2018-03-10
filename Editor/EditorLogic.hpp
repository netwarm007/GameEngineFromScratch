#pragma once
#include "IGameLogic.hpp"

namespace My {
    class EditorLogic : implements IGameLogic
    {
        int Initialize();
        void Finalize();
        void Tick();
        
        void OnLeftKeyDown();
        void OnRightKeyDown();
        void OnUpKeyDown();
        void OnDownKeyDown();

        void OnAnalogStick(int id, float deltaX, float deltaY);
#ifdef DEBUG
        void DrawDebugInfo();
#endif
    };
}
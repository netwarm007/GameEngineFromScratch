#pragma once
#include "IGameLogic.hpp"
#include "geommath.hpp"
#include "quickhull.hpp"

namespace My {
    class article_45_logic : public IGameLogic
    {
    public:
        int Initialize();
        void Finalize();
        void Tick();

        void OnLeftKeyDown();
        void OnRightKeyDown();
        void OnUpKeyDown();
        void OnDownKeyDown();
        void OnButton1Down();

        void OnAnalogStick(int id, float deltaX, float deltaY);
#ifdef DEBUG
        void DrawDebugInfo();
#endif

    private:
        QuickHull m_QuickHullA;
        QuickHull m_QuickHullB;
        bool m_bCollided = false;
    };
}
#pragma once
#include "IGameLogic.hpp"
#include "geommath.hpp"
#include "quickhull.hpp"

namespace My {
    class article_44_logic : public IGameLogic
    {
    public:
        int Initialize();
        void Finalize();
        void Tick();

        void OnLeftKey();
        void OnRightKey();
        void OnUpKey();
        void OnDownKey();

#ifdef DEBUG
        void DrawDebugInfo();
#endif

    private:
        QuickHull m_QuickHull;
    };
}
#pragma once
#ifdef DEBUG
#include "IRuntimeModule.hpp"

namespace My {
    class DebugManager : implements IRuntimeModule
    {
    public:
        int Initialize();
        void Finalize();
        void Tick();

        void ToggleDebugInfo();

        void DrawDebugInfo();

    protected:
        void DrawAxis();
        void DrawGrid();

        bool m_bDrawDebugInfo = true;
    };

    extern DebugManager* g_pDebugManager;
}

#endif
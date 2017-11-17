#pragma once
#include "Image.hpp"
#include "IRuntimeModule.hpp"

namespace My {
    class GraphicsManager : implements IRuntimeModule
    {
    public:
        virtual ~GraphicsManager() {}

        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

        virtual void Clear();
        virtual void Draw();
    };

    extern GraphicsManager* g_pGraphicsManager;
}


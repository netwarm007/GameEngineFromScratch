#pragma once
#include "Image.hpp"
#include "IRuntimeModule.hpp"
#include "Mesh.h"

namespace My {
    class GraphicsManager : implements IRuntimeModule
    {
    public:
        virtual ~GraphicsManager() {}

        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

        void DrawSingleMesh(const Mesh& mesh);
    };
}


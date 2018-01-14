#pragma once
#include "geommath.hpp"
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

    protected:
        struct DrawFrameContext {
            Matrix4X4f  m_worldMatrix;
            Matrix4X4f  m_viewMatrix;
            Matrix4X4f  m_projectionMatrix;
            Vector3f    m_lightPosition;
            Vector4f    m_lightColor;
        };

        DrawFrameContext    m_DrawFrameContext;
    };

    extern GraphicsManager* g_pGraphicsManager;
}


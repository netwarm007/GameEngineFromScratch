#pragma once
#include "IRuntimeModule.hpp"
#include "geommath.hpp"
#include "Image.hpp"
#include "Scene.hpp"

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

#ifdef DEBUG
        virtual void DrawPoint(const Point &point, const Vector3f& color);
        virtual void DrawPointSet(const PointSet &point_set, const Vector3f& color);
        virtual void DrawLine(const Vector3f &from, const Vector3f &to, const Vector3f &color);
        virtual void DrawBox(const Vector3f &bbMin, const Vector3f &bbMax, const Vector3f &color);
        virtual void ClearDebugBuffers();
#endif

    protected:
        virtual bool InitializeShaders();
        virtual void ClearShaders();
        virtual void InitializeBuffers(const Scene& scene);
        virtual void ClearBuffers();

        virtual void InitConstants();
        virtual void CalculateCameraMatrix();
        virtual void CalculateLights();
        virtual void RenderBuffers();

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


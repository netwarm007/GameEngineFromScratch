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

        virtual bool InitializeShader(const char* vsFilename, const char* fsFilename);
        virtual void ClearShaders();
        virtual void InitializeBuffers(const Scene& scene);
        virtual void ClearBuffers();

        virtual void Clear();
        virtual void Draw();

    protected:
        virtual bool SetPerFrameShaderParameters();
        virtual bool SetPerBatchShaderParameters(const char* paramName, const Matrix4X4f& param);
        virtual bool SetPerBatchShaderParameters(const char* paramName, const Vector3f& param);
        virtual bool SetPerBatchShaderParameters(const char* paramName, const float param);
        virtual bool SetPerBatchShaderParameters(const char* paramName, const int param);

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


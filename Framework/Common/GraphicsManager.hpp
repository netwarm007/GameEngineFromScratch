#pragma once
#include "geommath.hpp"
#include "Image.hpp"
#include "IRuntimeModule.hpp"
#include "geommath.hpp"

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
        bool SetPerFrameShaderParameters();
        bool SetPerBatchShaderParameters(const char* paramName, const Matrix4X4f& param);
        bool SetPerBatchShaderParameters(const char* paramName, const Vector3f& param);
        bool SetPerBatchShaderParameters(const char* paramName, const float param);
        bool SetPerBatchShaderParameters(const char* paramName, const int param);

        void InitConstants();
        bool InitializeShader(const char* vsFilename, const char* fsFilename);
        void InitializeBuffers();
        void CalculateCameraMatrix();
        void CalculateLights();
        void RenderBuffers();

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


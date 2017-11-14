#pragma once
#include "GraphicsManager.hpp"
#include "geommath.hpp"

namespace My {
    class OpenGLGraphicsManager : public GraphicsManager
    {
    public:
        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

        virtual void Clear();

        virtual void Draw();

    private:
        bool SetShaderParameters(float* worldMatrix, float* viewMatrix, float* projectionMatrix);

        bool InitializeBuffers();
        void RenderBuffers();
        void CalculateCameraPosition();
        bool InitializeShader(const char* vsFilename, const char* fsFilename);

    private:
        unsigned int m_vertexShader;
        unsigned int m_fragmentShader;
        unsigned int m_shaderProgram;

        const bool VSYNC_ENABLED = true;
        const float screenDepth = 1000.0f;
        const float screenNear = 0.1f;

        int     m_vertexCount, m_indexCount;
        unsigned int m_vertexArrayId, m_vertexBufferId, m_indexBufferId;

        float m_positionX = 0, m_positionY = 0, m_positionZ = -10;
        float m_rotationX = 0, m_rotationY = 0, m_rotationZ = 0;
        Matrix4X4f m_worldMatrix;
        Matrix4X4f m_viewMatrix;
        Matrix4X4f m_projectionMatrix;
    };
}


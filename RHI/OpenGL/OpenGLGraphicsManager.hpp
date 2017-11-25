#pragma once
#include "GraphicsManager.hpp"
#include "geommath.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include "glad/glad.h"

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

        void InitializeBuffers();
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

        struct DrawBatchContext {
            GLuint  vao;
            GLenum  mode;
            GLenum  type;
            GLsizei count;
        };

        std::vector<DrawBatchContext> m_VAO;
        std::unordered_map<std::string, unsigned int> m_Buffers;

        float m_positionX = 0, m_positionY = 0, m_positionZ = -10;
        float m_rotationX = 0, m_rotationY = 0, m_rotationZ = 0;
        Matrix4X4f m_worldMatrix;
        Matrix4X4f m_viewMatrix;
        Matrix4X4f m_projectionMatrix;
    };

}



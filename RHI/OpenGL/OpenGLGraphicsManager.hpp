#pragma once
#include "GraphicsManager.hpp"
#include "geommath.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
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
            std::shared_ptr<Matrix4X4f> transform;
        };

        std::vector<DrawBatchContext> m_VAO;
        std::unordered_map<std::string, unsigned int> m_Buffers;

        Matrix4X4f m_worldMatrix;
        Matrix4X4f m_viewMatrix;
        Matrix4X4f m_projectionMatrix;
    };

}



#pragma once
#include "glad/glad.h"
#include "IShaderManager.hpp"

namespace My {
    class OpenGLShaderManager : implements IShaderManager
    {
    public:
        OpenGLShaderManager() = default;
        ~OpenGLShaderManager() = default;

        virtual int Initialize() final;
        virtual void Finalize() final;

        virtual void Tick() final;

        virtual bool InitializeShaders() final;
        virtual void ClearShaders() final;

        virtual void* GetDefaultShaderProgram() final;

#ifdef DEBUG
        virtual void* GetDebugShaderProgram() final;
#endif

    private:
        GLuint m_vertexShader;
        GLuint m_fragmentShader;
        GLuint m_shaderProgram;
#ifdef DEBUG
        GLuint m_debugVertexShader;
        GLuint m_debugFragmentShader;
        GLuint m_debugShaderProgram;
#endif
    };
}
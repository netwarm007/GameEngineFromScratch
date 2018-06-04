#pragma once
#include "IShaderManager.hpp"

namespace My {
    class OpenGLShaderManagerCommonBase : implements IShaderManager
    {
    public:
        OpenGLShaderManagerCommonBase() = default;
        ~OpenGLShaderManagerCommonBase() = default;

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
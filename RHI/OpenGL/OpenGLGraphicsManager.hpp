#pragma once
#include "glad/glad.h"
#include "GraphicsManager.hpp"

namespace My {
    class OpenGLGraphicsManager : public GraphicsManager
    {
    public:
        virtual int Initialize();
        virtual void Finalize();

        virtual void Tick();

        bool InitializeShader(const char* vsFilename, const char* fsFilename);

    private:
        GLuint m_VertexShader;
        GLuint m_FragmentShader;
        GLuint m_ShaderProgram;
    };
}


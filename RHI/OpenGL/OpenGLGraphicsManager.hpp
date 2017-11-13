#pragma once
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
        unsigned int m_VertexShader;
        unsigned int m_FragmentShader;
        unsigned int m_ShaderProgram;
    };
}


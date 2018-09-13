#pragma once
#include "ShaderManager.hpp"

namespace My {
    class OpenGLShaderManagerCommonBase : public ShaderManager
    {
    public:
        OpenGLShaderManagerCommonBase() = default;
        ~OpenGLShaderManagerCommonBase() = default;

        virtual int Initialize() final;
        virtual void Finalize() final;

        virtual void Tick() final;

        virtual bool InitializeShaders() final;
        virtual void ClearShaders() final;
    };
}
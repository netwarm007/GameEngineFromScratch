#pragma once
#include "ShaderManager.hpp"

namespace My {
    class OpenGLShaderManagerCommonBase : public ShaderManager
    {
    public:
        OpenGLShaderManagerCommonBase() = default;
        ~OpenGLShaderManagerCommonBase() override = default;

        int Initialize() final;
        void Finalize() final;

        void Tick() final;

        bool InitializeShaders() final;
        void ClearShaders() final;
    };
}
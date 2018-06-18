#include <d3d12.h>
#include "ShaderManager.hpp"

namespace My {
    struct D3dShaderProgram {
        D3D12_SHADER_BYTECODE vertexShaderByteCode;
        D3D12_SHADER_BYTECODE pixelShaderByteCode;
    };

    class D3dShaderManager : public ShaderManager
    {
    public:
        D3dShaderManager() = default;
        ~D3dShaderManager() = default;

        virtual int Initialize() final;
        virtual void Finalize() final;

        virtual void Tick() final;

        virtual bool InitializeShaders() final;
        virtual void ClearShaders() final;
    };
}
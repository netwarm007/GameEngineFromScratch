#include "ShaderManager.hpp"
#include <unordered_map>
#include <portable.hpp>

OBJC_CLASS(MTLLibrary);
OBJC_CLASS(MTLRenderPipelineState);

namespace My {
    class MetalShaderManager : public ShaderManager
    {
    public:
        MetalShaderManager() = default;
        ~MetalShaderManager() = default;

        int Initialize() final;
        void Finalize() final;

        void Tick() final;

        bool InitializeShaders() final;
        void ClearShaders() final;

    private:
        MTLLibrary* m_shaderLibrary;
        std::unordered_map<ShaderHandler, MTLRenderPipelineState*> m_pipelineStates;
    };
}
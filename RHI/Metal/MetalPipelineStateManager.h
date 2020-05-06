#include "PipelineStateManager.hpp"
#include <portable.hpp>
#include <unordered_map>


OBJC_CLASS(MTLLibrary);

namespace My {
    class MetalPipelineStateManager : public PipelineStateManager
    {
    public:
        MetalPipelineStateManager() = default;
        ~MetalPipelineStateManager() override = default;

    protected:
        bool InitializePipelineState(PipelineState** ppPipelineState) final;
        void DestroyPipelineState(PipelineState& pipelineState) final;

    private:
        MTLLibrary* m_shaderLibrary;
    };
}
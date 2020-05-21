#include "D3d/D3d12GraphicsManager.hpp"
#include "D3d/D3d12PipelineStateManager.hpp"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new D3d12GraphicsManager);
IPipelineStateManager* g_pPipelineStateManager =
    static_cast<IPipelineStateManager*>(new D3d12PipelineStateManager);
}  // namespace My

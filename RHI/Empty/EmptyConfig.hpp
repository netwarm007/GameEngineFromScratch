#include "EmptyPipelineStateManager.hpp"
#include "GraphicsManager.hpp"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new GraphicsManager);
IPipelineStateManager* g_pPipelineStateManager =
    static_cast<IPipelineStateManager*>(new EmptyPipelineStateManager);
}  // namespace My

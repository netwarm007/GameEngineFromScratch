#pragma once
#include "Metal/Metal2GraphicsManager.h"
#include "Metal/MetalPipelineStateManager.h"

namespace My {
GraphicsManager* g_pGraphicsManager =
    static_cast<GraphicsManager*>(new Metal2GraphicsManager);
PipelineStateManager* g_pPipelineStateManager =
    static_cast<PipelineStateManager*>(new MetalPipelineStateManager);
}  // namespace My

#pragma once
#include "Metal/Metal2GraphicsManager.h"
#include "Metal/MetalShaderManager.h"

namespace My {
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new Metal2GraphicsManager);
    IShaderManager*  g_pShaderManager   = static_cast<IShaderManager*>(new MetalShaderManager);
}

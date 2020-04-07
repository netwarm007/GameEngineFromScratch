#include "Empty/EmptyShaderManager.hpp"
#include "GraphicsManager.hpp"

namespace My {
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new GraphicsManager);
    IShaderManager*  g_pShaderManager   = static_cast<IShaderManager*>(new EmptyShaderManager);
}

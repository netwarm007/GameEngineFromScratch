#include "D3d/D3d12GraphicsManager.hpp"
#include "D3d/D3dShaderManager.hpp"

namespace My {
    GraphicsManager* g_pGraphicsManager = static_cast<GraphicsManager*>(new D3d12GraphicsManager);
    IShaderManager*  g_pShaderManager   = static_cast<IShaderManager*>(new D3dShaderManager);
}

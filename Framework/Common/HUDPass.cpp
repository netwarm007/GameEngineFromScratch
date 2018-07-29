#include "HUDPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;
using namespace std;

void HUDPass::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Copy);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

#ifdef DEBUG
    // Draw Shadow Maps
    float top = 0.95f;
    float left = 0.60f;

    for (auto shadowmap : frame.shadowMaps)
    {
        g_pGraphicsManager->DrawOverlay(shadowmap.second, left, top, 0.35f, 0.35f);
        top -= 0.45f;
    }
#endif
}

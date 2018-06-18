#include "ForwardRenderPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;
using namespace std;

void ForwardRenderPass::Draw(const Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Forward);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    g_pGraphicsManager->SetPerFrameConstants(frame.frameContext);

    for (auto dbc : frame.batchContexts)
    {
        g_pGraphicsManager->DrawBatch(*dbc);
    }
}

#include "SkyBoxPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;

void SkyBoxPass::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::SkyBox);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    g_pGraphicsManager->SetPerFrameConstants(frame.frameContext);

    g_pGraphicsManager->SetSkyBox(frame.frameContext);
    g_pGraphicsManager->DrawSkyBox();
}

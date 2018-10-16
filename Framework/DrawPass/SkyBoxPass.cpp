#include "SkyBoxPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;

void SkyBoxPass::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::SkyBox);

    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    g_pGraphicsManager->SetPerFrameConstants(frame.frameContext);

    g_pGraphicsManager->SetSkyBox(frame.frameContext);
    g_pGraphicsManager->DrawSkyBox();
}

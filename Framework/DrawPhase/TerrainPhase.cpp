#include "TerrainPhase.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"

using namespace My;

void TerrainPhase::Draw(Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Terrain);

    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    g_pGraphicsManager->SetTerrain(frame.frameContext);
    g_pGraphicsManager->DrawTerrain();
}

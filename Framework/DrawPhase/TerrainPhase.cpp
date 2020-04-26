#include "TerrainPhase.hpp"
#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;

void TerrainPhase::Draw(Frame& frame)
{
    auto& pipelineState = g_pPipelineStateManager->GetPipelineState("Terrain");

    g_pGraphicsManager->SetPipelineState(pipelineState);

    g_pGraphicsManager->DrawTerrain();
}

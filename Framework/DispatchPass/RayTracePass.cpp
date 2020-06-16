#include "RayTracePass.hpp"

#include "GraphicsManager.hpp"
#include "IPipelineStateManager.hpp"

using namespace My;

static int32_t raytrace_texture = -1;

void RayTracePass::Dispatch(Frame& frame) {
    auto& pPipelineState =
        g_pPipelineStateManager->GetPipelineState("RAYTRACE");

    // Set the color shader as the current shader program and set the matrices
    // that it will use for rendering.
    g_pGraphicsManager->SetPipelineState(pPipelineState, frame);

    const uint32_t width = 512u;
    const uint32_t height = 512u;
    const uint32_t depth = 1u;
    if (raytrace_texture == -1) {
        g_pGraphicsManager->GenerateTextureForWrite("RAYTRACE", width, height);
        raytrace_texture = g_pGraphicsManager->GetTexture("RAYTRACE");
    }
    g_pGraphicsManager->BindTextureForWrite("RAYTRACE", 0);
    g_pGraphicsManager->Dispatch(width, height, depth);
}

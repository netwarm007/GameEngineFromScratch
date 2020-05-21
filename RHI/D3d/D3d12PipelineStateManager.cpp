#include "D3d12PipelineStateManager.hpp"

#include "AssetLoader.hpp"

using namespace My;
using namespace std;

#define SHADER_ROOT "Shaders/HLSL/"
#define SHADER_SUFFIX ".cso"

static void loadShaders(D3d12PipelineState* pState) {
    // load the shaders
    Buffer vertexShader, pixelShader, geometryShader, computeShader;
    if (!pState->vertexShaderName.empty()) {
        vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(
            (SHADER_ROOT + pState->vertexShaderName + SHADER_SUFFIX).c_str());
    }

    if (!pState->pixelShaderName.empty()) {
        pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(
            (SHADER_ROOT + pState->pixelShaderName + SHADER_SUFFIX).c_str());
    }

    if (!pState->geometryShaderName.empty()) {
        geometryShader = g_pAssetLoader->SyncOpenAndReadBinary(
            (SHADER_ROOT + pState->geometryShaderName + SHADER_SUFFIX).c_str());
    }

    if (!pState->computeShaderName.empty()) {
        computeShader = g_pAssetLoader->SyncOpenAndReadBinary(
            (SHADER_ROOT + pState->computeShaderName + SHADER_SUFFIX).c_str());
    }

    pState->vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();
    pState->vertexShaderByteCode.pShaderBytecode = vertexShader.MoveData();

    pState->pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();
    pState->pixelShaderByteCode.pShaderBytecode = pixelShader.MoveData();

    pState->geometryShaderByteCode.BytecodeLength =
        geometryShader.GetDataSize();
    pState->geometryShaderByteCode.pShaderBytecode = geometryShader.MoveData();

    pState->computeShaderByteCode.BytecodeLength = computeShader.GetDataSize();
    pState->computeShaderByteCode.pShaderBytecode = computeShader.MoveData();
}

bool D3d12PipelineStateManager::InitializePipelineState(
    PipelineState** ppPipelineState) {
    D3d12PipelineState* pState = new D3d12PipelineState(**ppPipelineState);

    loadShaders(pState);

    *ppPipelineState = pState;

    return true;
}

void D3d12PipelineStateManager::DestroyPipelineState(
    PipelineState& pipelineState) {
    // D3d12PipelineState* pPipelineState =
    // dynamic_cast<D3d12PipelineState*>(&pipelineState);
}

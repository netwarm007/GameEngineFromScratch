#include "D3dShaderManager.hpp"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

const char VS_SHADER_SOURCE_FILE[] = "Shaders/basic_vs.cso";
const char PS_SHADER_SOURCE_FILE[] = "Shaders/basic_ps.cso";
#ifdef DEBUG
const char DEBUG_VS_SHADER_SOURCE_FILE[] = "Shaders/debug_vs.cso";
const char DEBUG_PS_SHADER_SOURCE_FILE[] = "Shaders/debug_ps.cso";
#endif

int D3dShaderManager::Initialize()
{
    return InitializeShaders() == false;
}

void D3dShaderManager::Finalize()
{
    ClearShaders();
}

void D3dShaderManager::Tick()
{

}

bool D3dShaderManager::InitializeShaders()
{
    HRESULT hr = S_OK;
    const char* vsFilename = VS_SHADER_SOURCE_FILE;
    const char* fsFilename = PS_SHADER_SOURCE_FILE;
#ifdef DEBUG
    const char* debugVsFilename = DEBUG_VS_SHADER_SOURCE_FILE;
    const char* debugFsFilename = DEBUG_PS_SHADER_SOURCE_FILE;
#endif

    // load the shaders
    // forward rendering shader
    Buffer vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(vsFilename);
    Buffer pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(fsFilename);

    D3dShaderProgram& shaderProgram = *(new D3dShaderProgram);
    shaderProgram.vertexShaderByteCode.pShaderBytecode = vertexShader.GetData();
    shaderProgram.vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();

    shaderProgram.pixelShaderByteCode.pShaderBytecode = pixelShader.GetData();
    shaderProgram.pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();

    m_DefaultShaders[DefaultShaderIndex::Forward] = reinterpret_cast<intptr_t>(&shaderProgram);

#ifdef DEBUG
    // debug shader
    shaderProgram = *(new D3dShaderProgram);
    vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(debugVsFilename);
    pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(debugFsFilename);

    shaderProgram.vertexShaderByteCode.pShaderBytecode = vertexShader.GetData();
    shaderProgram.vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();

    shaderProgram.pixelShaderByteCode.pShaderBytecode = pixelShader.GetData();
    shaderProgram.pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();

    m_DefaultShaders[DefaultShaderIndex::Debug] = reinterpret_cast<intptr_t>(&shaderProgram);
#endif

    return hr == S_OK;
}

void D3dShaderManager::ClearShaders()
{
}

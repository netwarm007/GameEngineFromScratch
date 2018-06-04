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
    Buffer vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(vsFilename);
    Buffer pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(fsFilename);

    m_shaderProgram.vertexShaderByteCode.pShaderBytecode = vertexShader.GetData();
    m_shaderProgram.vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();

    m_shaderProgram.pixelShaderByteCode.pShaderBytecode = pixelShader.GetData();
    m_shaderProgram.pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();

    return hr == S_OK;
}

void D3dShaderManager::ClearShaders()
{
}

void* D3dShaderManager::GetDefaultShaderProgram()
{
    return static_cast<void*>(&m_shaderProgram);
}

#ifdef DEBUG
void* D3dShaderManager::GetDebugShaderProgram()
{
    return static_cast<void*>(&m_debugShaderProgram);
}
#endif

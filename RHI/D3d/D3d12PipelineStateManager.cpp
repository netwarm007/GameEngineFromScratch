#include "D3d12PipelineStateManager.hpp"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

#define SHADER_ROOT "Shaders/HLSL/"
#define SHADER_SUFFIX ".cso"

static void loadShaders(D3dShaderProgram& program, 
                        const char* vs, 
                        const char* ps, 
                        const char* gs = nullptr,
                        const char* cs = nullptr
                       )
{
    // load the shaders
    Buffer vertexShader, pixelShader, geometryShader, computeShader;
    if (vs)
    {
        vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(vs);
    }

    if (ps)
    {
        pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(ps);
    }

    if (gs)
    {
        geometryShader = g_pAssetLoader->SyncOpenAndReadBinary(gs);
    }

    if (cs)
    {
        computeShader = g_pAssetLoader->SyncOpenAndReadBinary(cs);
    }

    program.vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();
    program.vertexShaderByteCode.pShaderBytecode = vertexShader.MoveData();

    program.pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();
    program.pixelShaderByteCode.pShaderBytecode = pixelShader.MoveData();

    program.geometryShaderByteCode.BytecodeLength = geometryShader.GetDataSize();
    program.geometryShaderByteCode.pShaderBytecode = geometryShader.MoveData();

    program.computeShaderByteCode.BytecodeLength = computeShader.GetDataSize();
    program.computeShaderByteCode.pShaderBytecode = computeShader.MoveData();
}

bool D3d12PipelineStateManager::InitializeShaders()
{
    HRESULT hr = S_OK;

    // load the shaders
    // basic shader
    D3dShaderProgram* shaderProgram = new D3dShaderProgram;
    loadShaders(*shaderProgram, VS_BASIC_SOURCE_FILE, PS_BASIC_SOURCE_FILE);
    shaderProgram->a2vType = A2V_TYPES::A2V_TYPES_FULL;

    m_DefaultShaders[DefaultShaderIndex::Basic] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);

    // pbr shader
    shaderProgram = new D3dShaderProgram;
    loadShaders(*shaderProgram, VS_PBR_SOURCE_FILE, PS_PBR_SOURCE_FILE);
    shaderProgram->a2vType = A2V_TYPES::A2V_TYPES_FULL;

    m_DefaultShaders[DefaultShaderIndex::Pbr] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);

    // skybox shader
    shaderProgram = new D3dShaderProgram;
    loadShaders(*shaderProgram, VS_SKYBOX_SOURCE_FILE, PS_SKYBOX_SOURCE_FILE);
    shaderProgram->a2vType = A2V_TYPES::A2V_TYPES_CUBE;

    m_DefaultShaders[DefaultShaderIndex::SkyBox] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);

    // shadowmap shader
    shaderProgram = new D3dShaderProgram;
    loadShaders(*shaderProgram, VS_SHADOWMAP_SOURCE_FILE, PS_SHADOWMAP_SOURCE_FILE);
    shaderProgram->a2vType = A2V_TYPES::A2V_TYPES_POS_ONLY;

    m_DefaultShaders[DefaultShaderIndex::ShadowMap] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);

#if 0
    // omni shadowmap shader
    shaderProgram = new D3dShaderProgram;
    loadShaders(*shaderProgram, VS_OMNI_SHADOWMAP_SOURCE_FILE, PS_OMNI_SHADOWMAP_SOURCE_FILE, GS_OMNI_SHADOWMAP_SOURCE_FILE);
    shaderProgram->a2vType = A2V_TYPES::A2V_TYPES_POS_ONLY;

    m_DefaultShaders[DefaultShaderIndex::OmniShadowMap] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);
#endif

#ifdef DEBUG
    // debug shader
    shaderProgram = new D3dShaderProgram;
    loadShaders(*shaderProgram, DEBUG_VS_SHADER_SOURCE_FILE, DEBUG_PS_SHADER_SOURCE_FILE);

    m_DefaultShaders[DefaultShaderIndex::Debug] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);
#endif

    return hr == S_OK;
}

void D3d12PipelineStateManager::ClearShaders()
{
    for (auto& it : m_DefaultShaders)
    {
        D3dShaderProgram* shaderProgram = reinterpret_cast<D3dShaderProgram*>(it.second);
        if (shaderProgram)
        {
            delete shaderProgram;
        }
    }
}

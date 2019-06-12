#include "D3dShaderManager.hpp"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

#define SHADER_ROOT "Shaders/HLSL/"

#define VS_BASIC_SOURCE_FILE SHADER_ROOT "basic.vert.cso"
#define PS_BASIC_SOURCE_FILE SHADER_ROOT "basic.frag.cso"
#define VS_SHADOWMAP_SOURCE_FILE SHADER_ROOT "shadowmap.vert.cso"
#define PS_SHADOWMAP_SOURCE_FILE SHADER_ROOT "shadowmap.frag.cso"
#define VS_OMNI_SHADOWMAP_SOURCE_FILE SHADER_ROOT "shadowmap_omni.vert.cso"
#define PS_OMNI_SHADOWMAP_SOURCE_FILE SHADER_ROOT "shadowmap_omni.frag.cso"
#define GS_OMNI_SHADOWMAP_SOURCE_FILE SHADER_ROOT "shadowmap_omni.geom.cso"
#define DEBUG_VS_SHADER_SOURCE_FILE SHADER_ROOT "debug.vert.cso"
#define DEBUG_PS_SHADER_SOURCE_FILE SHADER_ROOT "debug.frag.cso"
#define VS_PASSTHROUGH_SOURCE_FILE SHADER_ROOT "passthrough.vert.cso"
#define PS_TEXTURE_SOURCE_FILE SHADER_ROOT "texture.frag.cso"
#define PS_TEXTURE_ARRAY_SOURCE_FILE SHADER_ROOT "texturearray.frag.cso"
#define VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE SHADER_ROOT "passthrough_cube.vert.cso"
#define PS_CUBEMAP_SOURCE_FILE SHADER_ROOT "cubemap.frag.cso"
#define PS_CUBEMAP_ARRAY_SOURCE_FILE SHADER_ROOT "cubemaparray.frag.cso"
#define VS_SKYBOX_SOURCE_FILE SHADER_ROOT "skybox.vert.cso"
#define PS_SKYBOX_SOURCE_FILE SHADER_ROOT "skybox.frag.cso"
#define VS_PBR_SOURCE_FILE SHADER_ROOT "pbr.vert.cso"
#define PS_PBR_SOURCE_FILE SHADER_ROOT "pbr.frag.cso"
#define CS_PBR_BRDF_SOURCE_FILE SHADER_ROOT "integrateBRDF.comp.cso"
#define VS_TERRAIN_SOURCE_FILE SHADER_ROOT "terrain.vert.cso"
#define PS_TERRAIN_SOURCE_FILE SHADER_ROOT "terrain.frag.cso"
#define TESC_TERRAIN_SOURCE_FILE SHADER_ROOT "terrain.tesc.cso"
#define TESE_TERRAIN_SOURCE_FILE SHADER_ROOT "terrain.tese.cso"

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

bool D3dShaderManager::InitializeShaders()
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

void D3dShaderManager::ClearShaders()
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

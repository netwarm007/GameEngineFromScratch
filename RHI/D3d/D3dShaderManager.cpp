#include "D3dShaderManager.hpp"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

#define VS_BASIC_SOURCE_FILE "Shaders/HLSL/basic.vert.cso"
#define PS_BASIC_SOURCE_FILE "Shaders/HLSL/basic.frag.cso"
#define VS_SHADOWMAP_SOURCE_FILE "Shaders/HLSL/shadowmap.vert.cso"
#define PS_SHADOWMAP_SOURCE_FILE "Shaders/HLSL/shadowmap.frag.cso"
#define VS_OMNI_SHADOWMAP_SOURCE_FILE "Shaders/HLSL/shadowmap_omni.vert.cso"
#define PS_OMNI_SHADOWMAP_SOURCE_FILE "Shaders/HLSL/shadowmap_omni.frag.cso"
#define GS_OMNI_SHADOWMAP_SOURCE_FILE "Shaders/HLSL/shadowmap_omni.geom.cso"
#define DEBUG_VS_SHADER_SOURCE_FILE "Shaders/HLSL/debug.vert.cso"
#define DEBUG_PS_SHADER_SOURCE_FILE "Shaders/HLSL/debug.frag.cso"
#define VS_PASSTHROUGH_SOURCE_FILE "Shaders/HLSL/passthrough.vert.cso"
#define PS_TEXTURE_SOURCE_FILE "Shaders/HLSL/texture.frag.cso"
#define PS_DEPTH_TEXTURE_ARRAY_SOURCE_FILE "Shaders/HLSL/depthtexturearray.frag.cso"
#define VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE "Shaders/HLSL/passthrough_cube.vert.cso"
#define PS_CUBEMAP_ARRAY_SOURCE_FILE "Shaders/HLSL/cubemaparray.frag.cso"
#define PS_DEPTH_CUBEMAP_ARRAY_SOURCE_FILE "Shaders/HLSL/depthcubemaparray.frag.cso"
#define PS_SIMPLE_CUBEMAP_SOURCE_FILE "Shaders/HLSL/cubemap.frag.cso"
#define VS_SKYBOX_SOURCE_FILE "Shaders/HLSL/skybox.vert.cso"
#define PS_SKYBOX_SOURCE_FILE "Shaders/HLSL/skybox.frag.cso"
#define PS_PBR_SOURCE_FILE "Shaders/HLSL/pbr.frag.cso"
#define CS_PBR_BRDF_SOURCE_FILE "Shaders/HLSL/integrateBRDF.comp.cso"
#define PS_PBR_BRDF_SOURCE_FILE "Shaders/HLSL/integrateBRDF.frag.cso"
#define VS_TERRAIN_SOURCE_FILE "Shaders/HLSL/terrain.vert.cso"
#define PS_TERRAIN_SOURCE_FILE "Shaders/HLSL/terrain.frag.cso"
#define TESC_TERRAIN_SOURCE_FILE "Shaders/HLSL/terrain.tesc.cso"
#define TESE_TERRAIN_SOURCE_FILE "Shaders/HLSL/terrain.tese.cso"


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
    const char* vsFilename = VS_BASIC_SOURCE_FILE;
    const char* fsFilename = PS_BASIC_SOURCE_FILE;
    const char* debugVsFilename = DEBUG_VS_SHADER_SOURCE_FILE;
    const char* debugFsFilename = DEBUG_PS_SHADER_SOURCE_FILE;

    // load the shaders
    // forward rendering shader
    Buffer vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(VS_BASIC_SOURCE_FILE);
    Buffer pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(PS_BASIC_SOURCE_FILE);

    D3dShaderProgram* shaderProgram = new D3dShaderProgram;
    shaderProgram->vertexShaderByteCode.pShaderBytecode = vertexShader.GetData();
    shaderProgram->vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();

    shaderProgram->pixelShaderByteCode.pShaderBytecode = pixelShader.GetData();
    shaderProgram->pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();

    m_DefaultShaders[DefaultShaderIndex::Basic] = reinterpret_cast<IShaderManager::ShaderHandler>(shaderProgram);

#ifdef DEBUG
    // debug shader
    shaderProgram = new D3dShaderProgram;
    vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(debugVsFilename);
    pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(debugFsFilename);

    shaderProgram->vertexShaderByteCode.pShaderBytecode = vertexShader.GetData();
    shaderProgram->vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();

    shaderProgram->pixelShaderByteCode.pShaderBytecode = pixelShader.GetData();
    shaderProgram->pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();

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

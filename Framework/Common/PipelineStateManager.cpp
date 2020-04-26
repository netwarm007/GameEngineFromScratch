#include "PipelineStateManager.hpp"

using namespace My;
using namespace std;

#define VS_BASIC_SOURCE_FILE "basic.vert"
#define PS_BASIC_SOURCE_FILE "basic.frag"
#define VS_PBR_SOURCE_FILE "pbr.vert"
#define PS_PBR_SOURCE_FILE "pbr.frag"
#define CS_PBR_BRDF_SOURCE_FILE "integrateBRDF.comp"
#define VS_SHADOWMAP_SOURCE_FILE "shadowmap.vert"
#define PS_SHADOWMAP_SOURCE_FILE "shadowmap.frag"
#define VS_OMNI_SHADOWMAP_SOURCE_FILE "shadowmap_omni.vert"
#define PS_OMNI_SHADOWMAP_SOURCE_FILE "shadowmap_omni.frag"
#define GS_OMNI_SHADOWMAP_SOURCE_FILE "shadowmap_omni.geom"
#define DEBUG_VS_SHADER_SOURCE_FILE "debug.vert"
#define DEBUG_PS_SHADER_SOURCE_FILE "debug.frag"
#define VS_PASSTHROUGH_SOURCE_FILE "passthrough.vert"
#define PS_TEXTURE_SOURCE_FILE "texture.frag"
#define PS_TEXTURE_ARRAY_SOURCE_FILE "texturearray.frag"
#define VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE "passthrough_cube.vert"
#define PS_CUBEMAP_SOURCE_FILE "cubemap.frag"
#define PS_CUBEMAP_ARRAY_SOURCE_FILE "cubemaparray.frag"
#define VS_SKYBOX_SOURCE_FILE "skybox.vert"
#define PS_SKYBOX_SOURCE_FILE "skybox.frag"
#define VS_TERRAIN_SOURCE_FILE "terrain.vert"
#define PS_TERRAIN_SOURCE_FILE "terrain.frag"
#define TESC_TERRAIN_SOURCE_FILE "terrain.tesc"
#define TESE_TERRAIN_SOURCE_FILE "terrain.tese"

PipelineStateManager::PipelineStateManager()
{
}

PipelineStateManager::~PipelineStateManager()
{
    Clear();
}

bool PipelineStateManager::RegisterPipelineState(PipelineState& pipelineState)
{
    PipelineState* pPipelineState;
    pPipelineState = &pipelineState;
    if (InitializePipelineState(&pPipelineState))
    {
        m_pipelineStates.emplace(pipelineState.pipelineStateName, pPipelineState);
        return true;
    }

    return false;
}

void PipelineStateManager::UnregisterPipelineState(PipelineState& pipelineState)
{
    const auto& it = m_pipelineStates.find(pipelineState.pipelineStateName);
    if (it != m_pipelineStates.end())
    {
        DestroyPipelineState(*it->second);
    }
    m_pipelineStates.erase(it);
}

void PipelineStateManager::Clear()
{
    for (auto it = m_pipelineStates.begin(); it != m_pipelineStates.end(); it++)
    {
        if (it != m_pipelineStates.end())
        {
            DestroyPipelineState(*it->second);
        }
        m_pipelineStates.erase(it);
    }

    assert(m_pipelineStates.empty());
}

const std::shared_ptr<PipelineState> PipelineStateManager::GetPipelineState(std::string name) const
{
    const auto& it = m_pipelineStates.find(name);
    if (it != m_pipelineStates.end())
    {
        return it->second;
    }
    else
    {
        assert(!m_pipelineStates.empty());
        return m_pipelineStates.begin()->second;
    }
}

int PipelineStateManager::Initialize()
{
    PipelineState pipelineState;
    pipelineState.vertexShaderName = VS_BASIC_SOURCE_FILE;
    pipelineState.pixelShaderName  = PS_BASIC_SOURCE_FILE;
    pipelineState.pipelineStateName = "BASIC";
    pipelineState.depthTestMode = DEPTH_TEST_MODE::LESS_EQUAL;
    pipelineState.stencilTestMode = STENCIL_TEST_MODE::NONE;
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_PBR_SOURCE_FILE;
    pipelineState.pixelShaderName  = PS_PBR_SOURCE_FILE;
    pipelineState.pipelineStateName = "PBR";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName.clear();
    pipelineState.pixelShaderName.clear();
    pipelineState.computeShaderName = CS_PBR_BRDF_SOURCE_FILE;
    pipelineState.pipelineStateName = "PBR BRDF CS";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_OMNI_SHADOWMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_OMNI_SHADOWMAP_SOURCE_FILE;
    pipelineState.geometryShaderName = GS_OMNI_SHADOWMAP_SOURCE_FILE;
    pipelineState.computeShaderName.clear(); 
    pipelineState.pipelineStateName = "Omni Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_SHADOWMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_SHADOWMAP_SOURCE_FILE;
    pipelineState.geometryShaderName.clear();
    pipelineState.pipelineStateName = "Spot Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Area Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Sun Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Sun Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_PASSTHROUGH_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_TEXTURE_SOURCE_FILE;
    pipelineState.pipelineStateName = "Texture Debug Output";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_PASSTHROUGH_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_TEXTURE_ARRAY_SOURCE_FILE;
    pipelineState.pipelineStateName = "Texture Array Debug Output";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_CUBEMAP_SOURCE_FILE;
    pipelineState.pipelineStateName = "CubeMap Debug Output";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_CUBEMAP_ARRAY_SOURCE_FILE;
    pipelineState.pipelineStateName = "CubeMap Array Debug Output";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_SKYBOX_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_SKYBOX_SOURCE_FILE;
    pipelineState.pipelineStateName = "SkyBox";
    RegisterPipelineState(pipelineState);

    pipelineState.vertexShaderName = VS_TERRAIN_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_TERRAIN_SOURCE_FILE;
    pipelineState.tessControlShaderName = TESC_TERRAIN_SOURCE_FILE;
    pipelineState.tessEvaluateShaderName = TESE_TERRAIN_SOURCE_FILE;
    pipelineState.pipelineStateName = "Terrain";
    RegisterPipelineState(pipelineState);

    return 0;
}

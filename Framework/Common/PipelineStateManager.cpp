#include "PipelineStateManager.hpp"

#include "IApplication.hpp"

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
#define CS_RAYTRACE_SOURCE_FILE "raytrace.comp"

PipelineStateManager::~PipelineStateManager() { Clear(); }

bool PipelineStateManager::RegisterPipelineState(PipelineState& pipelineState) {
    PipelineState* pPipelineState;
    pPipelineState = &pipelineState;
    if (InitializePipelineState(&pPipelineState)) {
        m_pipelineStates.emplace(pipelineState.pipelineStateName,
                                 pPipelineState);
        return true;
    }

    return false;
}

void PipelineStateManager::UnregisterPipelineState(
    PipelineState& pipelineState) {
    const auto& it = m_pipelineStates.find(pipelineState.pipelineStateName);
    if (it != m_pipelineStates.end()) {
        DestroyPipelineState(*it->second);
    }
    m_pipelineStates.erase(it);
}

void PipelineStateManager::Clear() {
    for (auto it = m_pipelineStates.begin(); it != m_pipelineStates.end();
         it++) {
        if (it != m_pipelineStates.end()) {
            DestroyPipelineState(*it->second);
        }
        m_pipelineStates.erase(it);
    }

    assert(m_pipelineStates.empty());

    cout << "Pipeline State Manager Clear has been called. " << endl;
}

const std::shared_ptr<PipelineState> PipelineStateManager::GetPipelineState(
    std::string name) const {
    const auto& it = m_pipelineStates.find(name);
    if (it != m_pipelineStates.end()) {
        return it->second;
    } else {
        assert(!m_pipelineStates.empty());
        return m_pipelineStates.begin()->second;
    }
}

int PipelineStateManager::Initialize() {
    PipelineState pipelineState;
    pipelineState.pipelineStateName = "BASIC";
    pipelineState.pipelineType = PIPELINE_TYPE::GRAPHIC;
    pipelineState.vertexShaderName = VS_BASIC_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_BASIC_SOURCE_FILE;
    pipelineState.depthTestMode = DEPTH_TEST_MODE::LESS_EQUAL;
    pipelineState.bDepthWrite = true;
    pipelineState.stencilTestMode = STENCIL_TEST_MODE::NONE;
    pipelineState.cullFaceMode = CULL_FACE_MODE::BACK;
    pipelineState.sampleCount = g_pApp->GetConfiguration().msaaSamples;
    pipelineState.a2vType = A2V_TYPES::A2V_TYPES_FULL;
    pipelineState.flag = PIPELINE_FLAG::NONE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "PBR";
    pipelineState.vertexShaderName = VS_PBR_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_PBR_SOURCE_FILE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "PBR BRDF CS";
    pipelineState.pipelineType = PIPELINE_TYPE::COMPUTE;
    pipelineState.vertexShaderName.clear();
    pipelineState.pixelShaderName.clear();
    pipelineState.computeShaderName = CS_PBR_BRDF_SOURCE_FILE;
    pipelineState.a2vType = A2V_TYPES::A2V_TYPES_NONE;
    pipelineState.flag = PIPELINE_FLAG::NONE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "RAYTRACE";
    pipelineState.computeShaderName = CS_RAYTRACE_SOURCE_FILE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Omni Light Shadow Map";
    pipelineState.pipelineType = PIPELINE_TYPE::GRAPHIC;
    pipelineState.vertexShaderName = VS_OMNI_SHADOWMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_OMNI_SHADOWMAP_SOURCE_FILE;
    pipelineState.geometryShaderName = GS_OMNI_SHADOWMAP_SOURCE_FILE;
    pipelineState.computeShaderName.clear();
#ifdef OS_MACOS
    // we flip the viewport so we need flip the culling mode too.
    pipelineState.cullFaceMode = CULL_FACE_MODE::BACK;
#else
    pipelineState.cullFaceMode = CULL_FACE_MODE::FRONT;
#endif
    pipelineState.pixelFormat = PIXEL_FORMAT::INVALID;
    pipelineState.sampleCount = 1;
    pipelineState.a2vType = A2V_TYPES::A2V_TYPES_POS_ONLY;
    pipelineState.flag = PIPELINE_FLAG::SHADOW;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Spot Light Shadow Map";
    pipelineState.vertexShaderName = VS_SHADOWMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_SHADOWMAP_SOURCE_FILE;
    pipelineState.geometryShaderName.clear();
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Area Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Sun Light Shadow Map";
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Texture Debug Output";
    pipelineState.vertexShaderName = VS_PASSTHROUGH_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_TEXTURE_SOURCE_FILE;
    pipelineState.cullFaceMode = CULL_FACE_MODE::BACK;
    pipelineState.a2vType = A2V_TYPES::A2V_TYPES_SIMPLE;
    pipelineState.depthTestMode = DEPTH_TEST_MODE::ALWAYS;
    pipelineState.bDepthWrite = true;
    pipelineState.pixelFormat = PIXEL_FORMAT::BGRA8UNORM;
    pipelineState.sampleCount = g_pApp->GetConfiguration().msaaSamples;
    pipelineState.flag = PIPELINE_FLAG::DEBUG_DRAW;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Texture Array Debug Output";
    pipelineState.vertexShaderName = VS_PASSTHROUGH_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_TEXTURE_ARRAY_SOURCE_FILE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "CubeMap Debug Output";
    pipelineState.vertexShaderName = VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_CUBEMAP_SOURCE_FILE;
    pipelineState.a2vType = A2V_TYPES::A2V_TYPES_CUBE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "CubeMap Array Debug Output";
    pipelineState.vertexShaderName = VS_PASSTHROUGH_CUBEMAP_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_CUBEMAP_ARRAY_SOURCE_FILE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "SkyBox";
    pipelineState.vertexShaderName = VS_SKYBOX_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_SKYBOX_SOURCE_FILE;
    pipelineState.depthTestMode = DEPTH_TEST_MODE::LESS_EQUAL;
    pipelineState.bDepthWrite = false;
    pipelineState.sampleCount = g_pApp->GetConfiguration().msaaSamples;
    pipelineState.flag = PIPELINE_FLAG::NONE;
    RegisterPipelineState(pipelineState);

    pipelineState.pipelineStateName = "Terrain";
    pipelineState.vertexShaderName = VS_TERRAIN_SOURCE_FILE;
    pipelineState.pixelShaderName = PS_TERRAIN_SOURCE_FILE;
    pipelineState.tessControlShaderName = TESC_TERRAIN_SOURCE_FILE;
    pipelineState.tessEvaluateShaderName = TESE_TERRAIN_SOURCE_FILE;
    pipelineState.depthTestMode = DEPTH_TEST_MODE::LESS_EQUAL;
    pipelineState.bDepthWrite = true;
    pipelineState.sampleCount = g_pApp->GetConfiguration().msaaSamples;
    pipelineState.a2vType = A2V_TYPES::A2V_TYPES_POS_ONLY;
    pipelineState.flag = PIPELINE_FLAG::NONE;
    RegisterPipelineState(pipelineState);

    cout << "Pipeline State Manager Initialized. [" << m_pipelineStates.size()
         << "]" << endl;

    return 0;
}

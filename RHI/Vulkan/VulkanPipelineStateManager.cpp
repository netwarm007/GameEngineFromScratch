#include "VulkanPipelineStateManager.hpp"

#include <stdexcept>

#include "AssetLoader.hpp"
#include "VulkanApplication.hpp"

int VulkanPipelineStateManager::Initialize() {
    const int result = PipelineStateManager::Initialize();
    if (result != 0) return result;

    PipelineState state;
    state.pipelineStateName = "Vulkan Scene Geometry";
    state.pipelineType = PIPELINE_TYPE::GRAPHIC;
    state.depthTestMode = DEPTH_TEST_MODE::LESS_EQUAL;
    state.bDepthWrite = true;
    state.cullFaceMode = CULL_FACE_MODE::BACK;
    state.sampleCount = m_pApp->GetConfiguration().msaaSamples;
    state.a2vType = A2V_TYPES::A2V_TYPES_FULL;
    state.flag = PIPELINE_FLAG::NONE;
    if (!RegisterPipelineState(state)) {
        throw std::runtime_error(
            "Failed to register the Vulkan scene geometry pipeline");
    }
    return 0;
}

bool VulkanPipelineStateManager::InitializePipelineState(
    PipelineState** ppPipelineState) {
    VulkanPipelineState* pState = new VulkanPipelineState(**ppPipelineState);

    const bool isSceneGeometry =
        pState->pipelineStateName == "Vulkan Scene Geometry";
    const bool isSkyBox = pState->pipelineStateName == "SkyBox";
    const bool isShadowMap =
        pState->pipelineStateName == "Omni Light Shadow Map" ||
        pState->pipelineStateName == "Spot Light Shadow Map" ||
        pState->pipelineStateName == "Area Light Shadow Map" ||
        pState->pipelineStateName == "Sun Light Shadow Map";
    const bool isOmniShadow =
        pState->pipelineStateName == "Omni Light Shadow Map";
    const bool isBRDFCompute =
        pState->pipelineStateName == "PBR BRDF CS";
    if (isSkyBox) {
        pState->a2vType = A2V_TYPES::A2V_TYPES_POS_ONLY;
    }
    if (isSceneGeometry || isSkyBox || isShadowMap || isBRDFCompute) {
        auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
        if (!app) {
            delete pState;
            throw std::logic_error(
                "Vulkan application is unavailable while loading shaders");
        }

        AssetLoader assetLoader;
        if (assetLoader.Initialize() != 0) {
            delete pState;
            throw std::runtime_error(
                "Failed to initialize the asset loader for Vulkan shaders");
        }

        auto vertexCode =
            isBRDFCompute
                ? Buffer{}
                : assetLoader.SyncOpenAndReadBinary(
                      isSceneGeometry
                          ? "Shaders/Vulkan/vulkan_scene.vert.spv"
                          : isSkyBox
                                ? "Shaders/Vulkan/vulkan_skybox.vert.spv"
                                : "Shaders/Vulkan/vulkan_shadow.vert.spv");
        Buffer fragmentCode;
        Buffer computeCode;
        if (isBRDFCompute) {
            computeCode = assetLoader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/vulkan_integrateBRDF.comp.spv");
        }
        if (isSceneGeometry || isSkyBox || isOmniShadow) {
            fragmentCode = assetLoader.SyncOpenAndReadBinary(
                isSceneGeometry
                    ? "Shaders/Vulkan/vulkan_scene.frag.spv"
                    : isSkyBox ? "Shaders/Vulkan/vulkan_skybox.frag.spv"
                               : "Shaders/Vulkan/vulkan_shadow.frag.spv");
        }
        if ((isBRDFCompute && computeCode.GetDataSize() == 0) ||
            (!isBRDFCompute && vertexCode.GetDataSize() == 0) ||
            ((isSceneGeometry || isSkyBox || isOmniShadow) &&
             fragmentCode.GetDataSize() == 0)) {
            delete pState;
            throw std::runtime_error(
                "Vulkan scene shader binaries are missing or empty");
        }

        auto& rhi = app->GetRHI();
        try {
            if (isBRDFCompute) {
                pState->computeShaderByteCode =
                    rhi.createShaderModule(computeCode);
            } else {
                pState->vertexShaderByteCode =
                    rhi.createShaderModule(vertexCode);
            }
            if (isSceneGeometry || isSkyBox || isOmniShadow) {
                pState->pixelShaderByteCode =
                    rhi.createShaderModule(fragmentCode);
            }
            if (isSceneGeometry) {
                pState->supportsSceneGeometry = true;
            } else if (isSkyBox) {
                pState->supportsSkyBox = true;
            } else if (isBRDFCompute) {
                pState->supportsCompute = true;
            } else if (isShadowMap) {
                pState->supportsShadowMap = true;
                pState->shadowMapUsesFragmentShader = isOmniShadow;
            }
        } catch (...) {
            auto device = rhi.GetDevice();
            if (pState->vertexShaderByteCode) {
                device.destroyShaderModule(pState->vertexShaderByteCode);
            }
            if (pState->computeShaderByteCode) {
                device.destroyShaderModule(pState->computeShaderByteCode);
            }
            delete pState;
            throw;
        }
    }

    *ppPipelineState = pState;

    return true;
}

void VulkanPipelineStateManager::UnregisterPipelineState(
    PipelineState& pipelineState) {
    const auto it = m_pipelineStates.find(pipelineState.pipelineStateName);
    if (it == m_pipelineStates.end()) {
        DestroyPipelineState(pipelineState);
        return;
    }

    DestroyPipelineState(*it->second);
    m_pipelineStates.erase(it);
}

std::shared_ptr<PipelineState>
VulkanPipelineStateManager::GetSceneGeometryPipelineState() const {
    const auto it = m_pipelineStates.find("Vulkan Scene Geometry");
    return it == m_pipelineStates.end() ? nullptr : it->second;
}

void VulkanPipelineStateManager::DestroyPipelineState(
    PipelineState& pipelineState) {
    VulkanPipelineState* pPipelineState =
        dynamic_cast<VulkanPipelineState*>(&pipelineState);

    if (!pPipelineState) return;

    if (auto* app = dynamic_cast<VulkanApplication*>(m_pApp)) {
        auto device = app->GetRHI().GetDevice();
        if (device) {
            if (pPipelineState->pipelineState) {
                device.destroyPipeline(pPipelineState->pipelineState);
            }
            if (pPipelineState->pipelineLayout) {
                device.destroyPipelineLayout(pPipelineState->pipelineLayout);
            }
            if (pPipelineState->descriptorSetLayout) {
                device.destroyDescriptorSetLayout(
                    pPipelineState->descriptorSetLayout);
            }
            if (pPipelineState->vertexShaderByteCode) {
                device.destroyShaderModule(pPipelineState->vertexShaderByteCode);
            }
            if (pPipelineState->pixelShaderByteCode) {
                device.destroyShaderModule(pPipelineState->pixelShaderByteCode);
            }
            if (pPipelineState->geometryShaderByteCode) {
                device.destroyShaderModule(
                    pPipelineState->geometryShaderByteCode);
            }
            if (pPipelineState->computeShaderByteCode) {
                device.destroyShaderModule(
                    pPipelineState->computeShaderByteCode);
            }
        }
    }

}

void VulkanPipelineStateManager::InvalidatePipelines() {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) return;

    auto device = app->GetRHI().GetDevice();
    if (!device) return;

    for (const auto& [name, state] : m_pipelineStates) {
        auto* vulkanState = dynamic_cast<VulkanPipelineState*>(state.get());
        if (vulkanState && vulkanState->pipelineState) {
            device.destroyPipeline(vulkanState->pipelineState);
            vulkanState->pipelineState = nullptr;
        }
    }
}

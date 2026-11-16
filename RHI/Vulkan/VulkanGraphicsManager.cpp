#include "VulkanGraphicsManager.hpp"
#include "VulkanApplication.hpp"
#include "BaseApplication.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

using namespace My;

namespace {
TextureHandler ToTextureHandler(VkImageView imageView) {
    return reinterpret_cast<TextureHandler>(imageView);
}

TextureHandler ToTextureHandler(VkImage image) {
    return reinterpret_cast<TextureHandler>(image);
}

void BuildVulkanPointShadowMatrices(Matrix4X4f (&matrices)[6],
                                    const Vector3f& position,
                                    const Matrix4X4f& projection) {
    const Vector3f directions[] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                   {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const Vector3f up[] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                           {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    for (size_t face = 0; face < 6; ++face) {
        Matrix4X4f view;
        BuildViewRHMatrix(view, position, position + directions[face],
                          up[face]);
        MatrixMultiply(matrices[face], view, projection);
    }
}

vk::CullModeFlags ToVulkanCullMode(CULL_FACE_MODE mode) {
    switch (mode) {
        case CULL_FACE_MODE::NONE:
            return vk::CullModeFlagBits::eNone;
        case CULL_FACE_MODE::FRONT:
            return vk::CullModeFlagBits::eFront;
        case CULL_FACE_MODE::BACK:
            return vk::CullModeFlagBits::eBack;
    }
    throw std::invalid_argument("Unsupported Vulkan cull mode");
}
}  // namespace

VulkanGraphicsManager::~VulkanGraphicsManager() {
}

int VulkanGraphicsManager::Initialize() {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app || m_SwapchainResources) {
        std::cerr << "[VulkanGraphicsManager] A VulkanApplication must be set "
                     "before initialization and the manager must not already "
                     "be initialized"
                  << std::endl;
        return -1;
    }

    int result = GraphicsManager::Initialize();
    if (result != 0) return result;

    auto& rhi = app->GetRHI();
    rhi.createUniformBuffers(m_PerFrameUniformBuffers,
                             sizeof(PerFrameConstants));
    rhi.createStorageBuffers(m_LightStorageBuffers, sizeof(LightInfo));
    rhi.createUniformBuffers(m_ShadowConstantsBuffers,
                             sizeof(ShadowMapConstants));
    m_MaterialSampler = rhi.createTextureSampler(false, 16.0f);
    auto createSolidTexture = [&rhi](const std::array<uint8_t, 4>& pixel) {
        Image image;
        image.Width = 1;
        image.Height = 1;
        image.bitcount = 32;
        image.bitdepth = 8;
        image.pitch = 4;
        image.data_size = 4;
        image.pixel_format = PIXEL_FORMAT::RGBA8;
        image.data = new uint8_t[4]{pixel[0], pixel[1], pixel[2], pixel[3]};
        return rhi.createTextureImage(image);
    };
    try {
        m_DefaultMaterialTexture =
            createSolidTexture({255, 255, 255, 255});
        m_DefaultNormalTexture =
            createSolidTexture({128, 128, 255, 255});
        Image fallbackEnvironment;
        fallbackEnvironment.Width = 2;
        fallbackEnvironment.Height = 2;
        fallbackEnvironment.bitcount = 32;
        fallbackEnvironment.bitdepth = 8;
        fallbackEnvironment.pitch = 8;
        fallbackEnvironment.data_size = 20;
        fallbackEnvironment.pixel_format = PIXEL_FORMAT::RGBA8;
        fallbackEnvironment.data = new uint8_t[20]{};
        fallbackEnvironment.mipmaps.emplace_back(2, 2, 8, 0, 16);
        fallbackEnvironment.mipmaps.emplace_back(1, 1, 4, 16, 4);
        for (size_t pixel = 0; pixel < 5; ++pixel) {
            fallbackEnvironment.data[pixel * 4 + 3] = 255;
        }
        std::vector<VulkanRHI::CubeArraySubresource> fallbackSubresources;
        fallbackSubresources.reserve(24);
        for (uint32_t layer = 0; layer < 12; ++layer) {
            for (uint32_t mip = 0; mip < 2; ++mip) {
                fallbackSubresources.push_back(
                    {&fallbackEnvironment, mip, mip, layer});
            }
        }
        m_DefaultEnvironmentTexture = rhi.createTextureCubeArray(
            2, 2, 2, 2, fallbackSubresources);
    } catch (...) {
        auto device = rhi.GetDevice();
        auto destroyTexture = [&device](VulkanRHI::Texture& texture) {
            if (texture.descriptor) device.destroyImageView(texture.descriptor);
            if (texture.image) device.destroyImage(texture.image);
            if (texture.heap) device.freeMemory(texture.heap);
            texture = {};
        };
        destroyTexture(m_DefaultMaterialTexture);
        destroyTexture(m_DefaultNormalTexture);
        destroyTexture(m_DefaultEnvironmentTexture);
        if (m_MaterialSampler) {
            rhi.GetDevice().destroySampler(m_MaterialSampler);
        }
        m_MaterialSampler = nullptr;
        throw;
    }

    auto resources = std::make_shared<SwapchainResources>();
    m_SwapchainResources = resources;
    auto* baseApp = dynamic_cast<BaseApplication*>(app);
    auto* pipelineStateManager =
        baseApp ? dynamic_cast<VulkanPipelineStateManager*>(
                      baseApp->GetPipelineStateManager())
                : nullptr;
    rhi.DestroySwapChainCB(
        [this, resources, app, pipelineStateManager]() {
        auto device = app->GetRHI().GetDevice();
        if (m_bImGuiInitialized) {
            clearImGuiTextures();
            ImGui_ImplVulkan_Shutdown();
            m_bImGuiInitialized = false;
        }
        if (pipelineStateManager) {
            pipelineStateManager->InvalidatePipelines();
        }
        destroyOffscreenFrames();
        if (resources->renderPass) {
            device.destroyRenderPass(resources->renderPass);
            resources->renderPass = nullptr;
        }
        if (resources->offscreenRenderPass) {
            device.destroyRenderPass(resources->offscreenRenderPass);
            resources->offscreenRenderPass = nullptr;
        }
    });
    rhi.CreateSwapChainCB([this, resources, app]() {
        auto& rhi = app->GetRHI();
        resources->renderPass = rhi.createRenderPass();
        resources->offscreenRenderPass = rhi.createOffscreenRenderPass();
        rhi.createColorResources();
        rhi.createDepthResources();
        rhi.createFramebuffers(resources->renderPass);
        initializeImGuiBackend();
    });
    rhi.CreateSwapChain();
    m_ShadowMapRenderPass = rhi.createDepthRenderPass();

    return 0;
}

void VulkanGraphicsManager::initializeImGuiBackend() {
    if (m_bImGuiInitialized || !ImGui::GetCurrentContext()) return;

    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app || !m_SwapchainResources ||
        !m_SwapchainResources->renderPass) {
        throw std::logic_error(
            "Vulkan resources are unavailable for ImGui initialization");
    }

    auto& rhi = app->GetRHI();
    const uint32_t imageCount = rhi.GetSwapChainImageCount();
    if (imageCount < 2) {
        throw std::runtime_error(
            "Vulkan swapchain has too few images for ImGui");
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_1;
    initInfo.Instance = static_cast<VkInstance>(rhi.GetInstance());
    initInfo.PhysicalDevice =
        static_cast<VkPhysicalDevice>(rhi.GetPhysicalDevice());
    initInfo.Device = static_cast<VkDevice>(rhi.GetDevice());
    initInfo.QueueFamily = rhi.GetGraphicsQueueFamily();
    initInfo.Queue = static_cast<VkQueue>(rhi.GetGraphicsQueue());
    initInfo.DescriptorPoolSize = 1000;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = imageCount;
    initInfo.PipelineInfoMain.RenderPass =
        static_cast<VkRenderPass>(m_SwapchainResources->renderPass);
    initInfo.PipelineInfoMain.MSAASamples =
        static_cast<VkSampleCountFlagBits>(rhi.GetMsaaSamples());

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("Failed to initialize the Vulkan ImGui backend");
    }
    m_bImGuiInitialized = true;
}

TextureHandler VulkanGraphicsManager::GetImGuiTextureHandle(
    const TextureBase& texture) {
    if (!texture.handler) return 0;
    if (!m_bImGuiInitialized) {
        throw std::logic_error(
            "Vulkan ImGui texture requested before backend initialization");
    }

    const auto existing = m_ImGuiTextures.find(texture.handler);
    if (existing != m_ImGuiTextures.end()) {
        return reinterpret_cast<TextureHandler>(
            static_cast<VkDescriptorSet>(existing->second));
    }

    const auto imageView = reinterpret_cast<VkImageView>(texture.handler);
    const auto imageLayout =
        texture.pixel_format == PIXEL_FORMAT::D32
            ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
            : vk::ImageLayout::eShaderReadOnlyOptimal;
    const auto descriptorSet = ImGui_ImplVulkan_AddTexture(
        imageView, static_cast<VkImageLayout>(imageLayout));
    if (!descriptorSet) {
        throw std::runtime_error(
            "Failed to create a Vulkan ImGui texture descriptor");
    }
    try {
        m_ImGuiTextures.emplace(texture.handler,
                                static_cast<vk::DescriptorSet>(descriptorSet));
    } catch (...) {
        ImGui_ImplVulkan_RemoveTexture(descriptorSet);
        throw;
    }
    return reinterpret_cast<TextureHandler>(descriptorSet);
}

void VulkanGraphicsManager::removeImGuiTexture(TextureHandler texture) {
    const auto it = m_ImGuiTextures.find(texture);
    if (it == m_ImGuiTextures.end()) return;
    if (m_bImGuiInitialized) {
        ImGui_ImplVulkan_RemoveTexture(
            static_cast<VkDescriptorSet>(it->second));
    }
    m_ImGuiTextures.erase(it);
}

void VulkanGraphicsManager::clearImGuiTextures() {
    if (m_bImGuiInitialized) {
        for (const auto& [texture, descriptorSet] : m_ImGuiTextures) {
            ImGui_ImplVulkan_RemoveTexture(
                static_cast<VkDescriptorSet>(descriptorSet));
        }
    }
    m_ImGuiTextures.clear();
}

void VulkanGraphicsManager::Finalize() {
    GraphicsManager::Finalize();

    if (m_SwapchainResources) {
        auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
        if (!app) {
            throw std::logic_error(
                "Vulkan application is unavailable while releasing the "
                "swapchain resources");
        }
        auto& rhi = app->GetRHI();
        auto device = rhi.GetDevice();
        if (device) {
            device.waitIdle();
            rhi.cleanupSwapChainResources();
            destroyUniformBuffers(m_PerFrameUniformBuffers);
            destroyUniformBuffers(m_LightStorageBuffers);
            destroyUniformBuffers(m_ShadowConstantsBuffers);
            destroyShadowMapArrays();
            if (m_ShadowMapRenderPass) {
                device.destroyRenderPass(m_ShadowMapRenderPass);
                m_ShadowMapRenderPass = nullptr;
            }
            if (m_MaterialSampler) {
                device.destroySampler(m_MaterialSampler);
                m_MaterialSampler = nullptr;
            }
            auto destroyTexture = [&device](VulkanRHI::Texture& texture) {
                if (texture.descriptor) device.destroyImageView(texture.descriptor);
                if (texture.image) device.destroyImage(texture.image);
                if (texture.heap) device.freeMemory(texture.heap);
                texture = {};
            };
            destroyTexture(m_DefaultMaterialTexture);
            destroyTexture(m_DefaultNormalTexture);
            destroyTexture(m_DefaultEnvironmentTexture);
        }
        rhi.CreateSwapChainCB({});
        rhi.DestroySwapChainCB({});
    }

    m_SwapchainResources.reset();
}

size_t VulkanGraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
    const void* pData = v_property_array.GetData();
    const auto size = v_property_array.GetDataSize();
    const auto vertexCount = v_property_array.GetVertexCount();
    if (!pData || size == 0 || vertexCount == 0 || size % vertexCount != 0) {
        throw std::invalid_argument(
            "Vulkan vertex buffers require non-empty, valid vertex data");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    m_VertexBuffers.reserve(m_VertexBuffers.size() + 1);
    VulkanRHI::VertexBuffer buffer{};
    rhi.createVertexBuffer(pData, size, buffer);
    m_VertexBuffers.push_back(buffer);

    return m_VertexBuffers.size() - 1;
}

size_t VulkanGraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
    const void* pData = index_array.GetData();
    const auto indexCount = index_array.GetIndexCount();
    const auto size = index_array.GetDataSize();
    if (!pData || indexCount == 0 ||
        indexCount > (std::numeric_limits<uint32_t>::max)()) {
        throw std::invalid_argument(
            "Vulkan index buffers require non-empty 16-bit or 32-bit data");
    }

    std::vector<uint32_t> expandedIndices;
    const void* uploadData = pData;
    size_t uploadSize = size;
    if (index_array.GetIndexType() == IndexDataType::kIndexDataTypeInt16) {
        if (size % sizeof(uint16_t) != 0 ||
            size / sizeof(uint16_t) != indexCount) {
            throw std::invalid_argument(
                "Vulkan 16-bit index buffer size does not match its count");
        }
        const auto* indices = static_cast<const uint16_t*>(pData);
        expandedIndices.assign(indices, indices + indexCount);
        uploadData = expandedIndices.data();
        uploadSize = expandedIndices.size() * sizeof(uint32_t);
    } else if (index_array.GetIndexType() !=
                   IndexDataType::kIndexDataTypeInt32 ||
               size % sizeof(uint32_t) != 0 ||
               size / sizeof(uint32_t) != indexCount) {
        throw std::invalid_argument(
            "Vulkan index buffers require 16-bit or 32-bit index data");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    m_IndexBuffers.reserve(m_IndexBuffers.size() + 1);
    VulkanRHI::IndexBuffer buffer{};
    rhi.createIndexBuffer(uploadData, uploadSize, buffer);
    m_IndexBuffers.push_back(buffer);

    return m_IndexBuffers.size() - 1;
}

void VulkanGraphicsManager::initializeGeometries(const Scene& scene) {
    std::cout << "Creating Draw Batch Contexts ...";
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    uint32_t batch_index = 0;
    for (const auto& _it : scene.GeometryNodes) {
        const auto& pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible()) {
            const auto& pGeometry =
                scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            if (!pGeometry) {
                throw std::runtime_error(
                    "Visible Vulkan geometry node has no scene object");
            }
            const auto& pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            const auto vertexPropertiesCount =
                pMesh->GetVertexPropertiesCount();
            if (vertexPropertiesCount == 0 || pMesh->GetIndexGroupCount() == 0) {
                throw std::runtime_error(
                    "Vulkan geometry requires vertex and index data");
            }

            const auto propertyOffset = m_VertexBuffers.size();
            size_t positionBufferOffset = 0;
            size_t normalBufferOffset = 0;
            size_t texcoordBufferOffset = 0;
            size_t tangentBufferOffset = 0;
            bool foundPosition = false;
            bool foundNormal = false;
            bool foundTexcoord = false;
            bool foundTangent = false;
            for (uint32_t i = 0; i < vertexPropertiesCount; i++) {
                const SceneObjectVertexArray& v_property_array =
                    pMesh->GetVertexPropertyArray(i);
                if (v_property_array.GetVertexCount() != pMesh->GetVertexCount()) {
                    throw std::runtime_error(
                        "Vulkan geometry vertex properties have mismatched "
                        "vertex counts");
                }
                if (v_property_array.GetAttributeName() == "position" ||
                    v_property_array.GetAttributeName() == "POSITION") {
                    if (v_property_array.GetDataType() !=
                        VertexDataType::kVertexDataTypeFloat3) {
                        throw std::runtime_error(
                            "Vulkan scene geometry currently requires float3 "
                            "position attributes");
                    }
                    positionBufferOffset = propertyOffset + i;
                    foundPosition = true;
                }
                if (v_property_array.GetAttributeName() == "normal" ||
                    v_property_array.GetAttributeName() == "NORMAL") {
                    if (v_property_array.GetDataType() !=
                        VertexDataType::kVertexDataTypeFloat3) {
                        throw std::runtime_error(
                            "Vulkan scene geometry currently requires float3 "
                            "normal attributes");
                    }
                    normalBufferOffset = propertyOffset + i;
                    foundNormal = true;
                }
                if (v_property_array.GetAttributeName() == "texcoord" ||
                    v_property_array.GetAttributeName() == "TEXCOORD" ||
                    v_property_array.GetAttributeName() == "uv" ||
                    v_property_array.GetAttributeName() == "UV") {
                    if (v_property_array.GetDataType() !=
                        VertexDataType::kVertexDataTypeFloat2) {
                        throw std::runtime_error(
                            "Vulkan scene geometry currently requires float2 "
                            "texture coordinates");
                    }
                    texcoordBufferOffset = propertyOffset + i;
                    foundTexcoord = true;
                }
                if (v_property_array.GetAttributeName() == "tangent" ||
                    v_property_array.GetAttributeName() == "TANGENT") {
                    if (v_property_array.GetDataType() !=
                        VertexDataType::kVertexDataTypeFloat3) {
                        throw std::runtime_error(
                            "Vulkan scene geometry currently requires float3 "
                            "tangent attributes");
                    }
                    tangentBufferOffset = propertyOffset + i;
                    foundTangent = true;
                }
                CreateVertexBuffer(v_property_array);
            }
            if (!foundPosition) {
                throw std::runtime_error(
                    "Vulkan scene geometry has no position attribute");
            }
            if (!foundTexcoord) {
                std::vector<float> texcoords(pMesh->GetVertexCount() * 2, 0.0f);
                auto* texcoordBytes =
                    new uint8_t[texcoords.size() * sizeof(float)];
                std::memcpy(texcoordBytes, texcoords.data(),
                            texcoords.size() * sizeof(float));
                SceneObjectVertexArray texcoordArray(
                    "texcoord", 0, VertexDataType::kVertexDataTypeFloat2,
                    texcoordBytes, texcoords.size());
                texcoordBufferOffset = CreateVertexBuffer(texcoordArray);
            }
            auto createDefaultAttribute = [&](const std::string& name,
                                              const Vector3f& value) {
                std::vector<float> values(pMesh->GetVertexCount() * 3);
                for (size_t vertex = 0; vertex < pMesh->GetVertexCount();
                     ++vertex) {
                    for (size_t component = 0; component < 3; ++component) {
                        values[vertex * 3 + component] = value[component];
                    }
                }
                auto* bytes = new uint8_t[values.size() * sizeof(float)];
                std::memcpy(bytes, values.data(),
                            values.size() * sizeof(float));
                SceneObjectVertexArray attribute(
                    name.c_str(), 0, VertexDataType::kVertexDataTypeFloat3,
                    bytes, values.size());
                return CreateVertexBuffer(attribute);
            };
            if (!foundNormal) {
                normalBufferOffset =
                    createDefaultAttribute("normal", {0.0f, 0.0f, 1.0f});
            }
            if (!foundTangent) {
                tangentBufferOffset =
                    createDefaultAttribute("tangent", {1.0f, 0.0f, 0.0f});
            }

            for (size_t group = 0; group < pMesh->GetIndexGroupCount(); ++group) {
                const SceneObjectIndexArray& index_array =
                    pMesh->GetIndexArray(group);
                auto dbc = std::make_shared<VulkanDrawBatchContext>();
                dbc->property_offset = propertyOffset;
                dbc->index_offset = CreateIndexBuffer(index_array);
                dbc->batchIndex = batch_index++;
                dbc->index_count = m_IndexBuffers[dbc->index_offset].indexCount;
                dbc->position_buffer_offset = positionBufferOffset;
                dbc->normal_buffer_offset = normalBufferOffset;
                dbc->texcoord_buffer_offset = texcoordBufferOffset;
                dbc->tangent_buffer_offset = tangentBufferOffset;
                dbc->property_count = vertexPropertiesCount;
                dbc->node = pGeometryNode;
                dbc->materialImageViews = {
                    m_DefaultMaterialTexture.descriptor,
                    m_DefaultNormalTexture.descriptor,
                    m_DefaultMaterialTexture.descriptor,
                    m_DefaultMaterialTexture.descriptor,
                    m_DefaultMaterialTexture.descriptor};
                const auto materialKey =
                    pGeometryNode->GetMaterialRef(index_array.GetMaterialIndex());
                const auto materialIt = scene.Materials.find(materialKey);
                if (materialIt != scene.Materials.end() && materialIt->second) {
                    const auto& material = *materialIt->second;
                    const auto& baseColor =
                        material.GetBaseColor();
                    dbc->baseColor = baseColor.Value;
                    auto uploadMaterialTexture =
                        [&](const std::shared_ptr<SceneObjectTexture>& texture,
                            const VulkanRHI::Texture& fallback) {
                            if (!texture) return fallback.descriptor;
                            auto textureIt =
                                m_MaterialTextures.find(texture.get());
                            if (textureIt == m_MaterialTextures.end()) {
                                auto image = texture->GetTextureImage();
                                if (!image || !image->data || !image->Width ||
                                    !image->Height) {
                                    throw std::runtime_error(
                                        "Vulkan material texture has no "
                                        "decoded image data: " +
                                        texture->GetName());
                                }
                                Image convertedImage;
                                Image* uploadImage = image.get();
                                if (!image->compressed) {
                                    const size_t channels =
                                        image->pixel_format ==
                                                PIXEL_FORMAT::RGB8
                                            ? 3
                                            : image->pixel_format ==
                                                      PIXEL_FORMAT::RGBA8
                                                  ? 4
                                                  : 0;
                                    const size_t rowBytes =
                                        static_cast<size_t>(image->Width) *
                                        channels;
                                    if (channels == 0 || image->bitdepth != 8 ||
                                        image->bitcount != channels * 8 ||
                                        image->pitch < rowBytes ||
                                        image->pitch == 0 ||
                                        image->data_size / image->pitch <
                                            image->Height) {
                                        throw std::runtime_error(
                                            "Vulkan material textures require "
                                            "8-bit RGB/RGBA or GPU-compressed "
                                            "images: " +
                                            texture->GetName());
                                    }

                                    if (channels != 4 ||
                                        image->pitch != rowBytes) {
                                        convertedImage.Width = image->Width;
                                        convertedImage.Height = image->Height;
                                        convertedImage.bitcount = 32;
                                        convertedImage.bitdepth = 8;
                                        convertedImage.pitch = rowBytes / channels * 4;
                                        convertedImage.data_size =
                                            convertedImage.pitch *
                                            convertedImage.Height;
                                        convertedImage.pixel_format =
                                            PIXEL_FORMAT::RGBA8;
                                        convertedImage.data =
                                            new uint8_t[convertedImage.data_size];
                                        for (size_t y = 0;
                                             y < convertedImage.Height; ++y) {
                                            for (size_t x = 0;
                                                 x < convertedImage.Width;
                                                 ++x) {
                                                const auto* source =
                                                    image->data +
                                                    y * image->pitch +
                                                    x * channels;
                                                auto* destination =
                                                    convertedImage.data +
                                                    y * convertedImage.pitch +
                                                    x * 4;
                                                std::memcpy(destination, source,
                                                            channels);
                                                if (channels == 3) {
                                                    destination[3] = 255;
                                                }
                                            }
                                        }
                                        uploadImage = &convertedImage;
                                    }
                                }
                                textureIt = m_MaterialTextures
                                                .emplace(
                                                    texture.get(),
                                                    rhi.createTextureImage(
                                                        *uploadImage))
                                                .first;
                            }
                            return textureIt->second.descriptor;
                        };
                    dbc->materialImageViews[0] = uploadMaterialTexture(
                        baseColor.ValueMap, m_DefaultMaterialTexture);
                    dbc->materialImageViews[1] = uploadMaterialTexture(
                        material.GetNormal().ValueMap, m_DefaultNormalTexture);
                    dbc->materialImageViews[2] = uploadMaterialTexture(
                        material.GetMetallic().ValueMap,
                        m_DefaultMaterialTexture);
                    dbc->materialImageViews[3] = uploadMaterialTexture(
                        material.GetRoughness().ValueMap,
                        m_DefaultMaterialTexture);
                    dbc->materialImageViews[4] = uploadMaterialTexture(
                        material.GetAO().ValueMap, m_DefaultMaterialTexture);

                    dbc->materialParameters = {
                        material.GetMetallic().Value,
                        material.GetRoughness().Value,
                        material.GetAO().Value, 0.0f};
                    if (!material.GetRoughness().ValueMap &&
                        dbc->materialParameters[1] == 0.0f) {
                        dbc->materialParameters[1] = 1.0f;
                    }
                    if (!material.GetAO().ValueMap &&
                        dbc->materialParameters[2] == 0.0f) {
                        dbc->materialParameters[2] = 1.0f;
                    }
                    if (baseColor.ValueMap && dbc->baseColor[0] == 0.0f &&
                        dbc->baseColor[1] == 0.0f &&
                        dbc->baseColor[2] == 0.0f &&
                        dbc->baseColor[3] == 0.0f) {
                        dbc->baseColor =
                            Vector4f{1.0f, 1.0f, 1.0f, 1.0f};
                    }
                }
                for (auto& frame : m_Frames) {
                    frame.batchContexts.push_back(dbc);
                }
            }
        }
    }

    m_PerBatchUniformBuffers.resize(m_Frames[0].batchContexts.size());
    for (size_t i = 0; i < m_PerBatchUniformBuffers.size(); ++i) {
        rhi.createUniformBuffers(m_PerBatchUniformBuffers[i],
                                 sizeof(VulkanSceneBatchConstants));
    }

    std::cout << "Done!" << std::endl;
}

void VulkanGraphicsManager::initializeSkyBox(const Scene& scene) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    assert(scene.SkyBox);
    std::array<std::shared_ptr<Image>, 18> images;
    std::array<Image, 18> convertedImages;
    std::array<const Image*, 18> uploadImages{};
    bool hasImage = false;
    for (size_t index = 0; index < images.size(); ++index) {
        images[index] = scene.SkyBox->GetTexture(index).GetTextureImage();
        hasImage = hasImage || static_cast<bool>(images[index]);
    }
    if (!hasImage) return;
    for (size_t index = 0; index < images.size(); ++index) {
        if (!images[index] || !images[index]->data ||
            images[index]->mipmaps.empty()) {
            throw std::runtime_error(
                "Vulkan skybox texture is missing decoded mip data: " +
                scene.SkyBox->GetTexture(index).GetName());
        }
        uploadImages[index] = images[index].get();
        const auto& source = *images[index];
        if (!source.compressed && source.pixel_format == PIXEL_FORMAT::RGB8) {
            if (source.bitcount != 24 || source.bitdepth != 8) {
                throw std::invalid_argument(
                    "Vulkan skybox RGB textures must use 8-bit channels");
            }
            auto& destination = convertedImages[index];
            destination.Width = source.Width;
            destination.Height = source.Height;
            destination.bitcount = 32;
            destination.bitdepth = 8;
            destination.pixel_format = PIXEL_FORMAT::RGBA8;
            for (const auto& mip : source.mipmaps) {
                const size_t sourceRowBytes =
                    static_cast<size_t>(mip.Width) * 3;
                if (mip.pitch < sourceRowBytes ||
                    mip.offset > source.data_size ||
                    mip.data_size > source.data_size - mip.offset ||
                    mip.data_size < mip.pitch * mip.Height) {
                    throw std::invalid_argument(
                        "Vulkan skybox RGB mip data is truncated");
                }
                const size_t destinationPitch =
                    static_cast<size_t>(mip.Width) * 4;
                const size_t destinationSize =
                    destinationPitch * mip.Height;
                const size_t destinationOffset = destination.data_size;
                if (destinationSize >
                    (std::numeric_limits<size_t>::max)() -
                        destinationOffset) {
                    throw std::length_error(
                        "Vulkan skybox RGB texture is too large");
                }
                destination.mipmaps.emplace_back(
                    mip.Width, mip.Height, destinationPitch,
                    destinationOffset, destinationSize);
                destination.data_size += destinationSize;
            }
            destination.pitch =
                static_cast<size_t>(destination.Width) * 4;
            destination.data = new uint8_t[destination.data_size];
            for (size_t mipIndex = 0;
                 mipIndex < source.mipmaps.size(); ++mipIndex) {
                const auto& sourceMip = source.mipmaps[mipIndex];
                const auto& destinationMip =
                    destination.mipmaps[mipIndex];
                for (uint32_t y = 0; y < sourceMip.Height; ++y) {
                    const auto* sourceRow =
                        source.data + sourceMip.offset + y * sourceMip.pitch;
                    auto* destinationRow =
                        destination.data + destinationMip.offset +
                        y * destinationMip.pitch;
                    for (uint32_t x = 0; x < sourceMip.Width; ++x) {
                        std::memcpy(destinationRow + x * 4,
                                    sourceRow + x * 3, 3);
                        destinationRow[x * 4 + 3] = 255;
                    }
                }
            }
            uploadImages[index] = &destination;
        }
    }

    const auto& environment = *images[0];
    if (environment.Width < 2 || environment.Width != environment.Height) {
        throw std::invalid_argument(
            "Vulkan skybox environment faces must be square and at least 2x2");
    }
    const uint32_t mipLevels = static_cast<uint32_t>(
        std::max<size_t>(2, images[12]->mipmaps.size()));
    if (mipLevels > std::bit_width(environment.Width)) {
        throw std::invalid_argument(
            "Vulkan skybox radiance map has too many mip levels");
    }

    std::vector<VulkanRHI::CubeArraySubresource> subresources;
    for (uint32_t face = 0; face < 6; ++face) {
        const auto& skyFace = *uploadImages[face];
        const auto& irradianceFace = *uploadImages[face + 6];
        if (skyFace.Width != environment.Width ||
            skyFace.Height != environment.Height ||
            irradianceFace.Width != environment.Width / 2 ||
            irradianceFace.Height != environment.Height / 2) {
            throw std::invalid_argument(
                "Vulkan skybox environment and irradiance face dimensions "
                "are inconsistent");
        }
        subresources.push_back({uploadImages[face], 0, 0, face});
        subresources.push_back({uploadImages[face + 6], 0, 1, face});
    }
    for (uint32_t face = 0; face < 6; ++face) {
        const auto& radianceFace = *uploadImages[face + 12];
        if (radianceFace.Width != environment.Width ||
            radianceFace.Height != environment.Height ||
            radianceFace.mipmaps.size() < mipLevels) {
            throw std::invalid_argument(
                "Vulkan skybox radiance faces have inconsistent dimensions "
                "or mip levels");
        }
        for (uint32_t mip = 0; mip < mipLevels; ++mip) {
            subresources.push_back(
                {uploadImages[face + 12], mip, mip, face + 6});
        }
    }

    m_SkyBoxTexture = rhi.createTextureCubeArray(
        environment.Width, environment.Height, mipLevels, 2, subresources);
    try {
        m_SkyBoxSampler =
            rhi.createTextureSampler(true, static_cast<float>(mipLevels - 1));
    } catch (...) {
        auto device = rhi.GetDevice();
        if (m_SkyBoxTexture.descriptor) {
            device.destroyImageView(m_SkyBoxTexture.descriptor);
        }
        if (m_SkyBoxTexture.image) device.destroyImage(m_SkyBoxTexture.image);
        if (m_SkyBoxTexture.heap) device.freeMemory(m_SkyBoxTexture.heap);
        m_SkyBoxTexture = {};
        throw;
    }

    auto& skybox = m_Frames[0].skybox;
    skybox.handler = ToTextureHandler(m_SkyBoxTexture.descriptor);
    skybox.buff = ToTextureHandler(m_SkyBoxTexture.image);
    skybox.format = static_cast<TextureFormat>(
        static_cast<VkFormat>(m_SkyBoxTexture.format));
    skybox.pixel_format = uploadImages[0]->pixel_format;
    skybox.width = environment.Width;
    skybox.height = environment.Height;
    skybox.mips = mipLevels;
    skybox.size = 2;
    m_Textures.push_back(skybox);

    std::array<uint32_t, SceneObjectSkyBox::skyboxIndices.size()> indices{};
    std::transform(SceneObjectSkyBox::skyboxIndices.begin(),
                   SceneObjectSkyBox::skyboxIndices.end(), indices.begin(),
                   [](uint16_t index) { return static_cast<uint32_t>(index); });
    auto* vertexBytes =
        new uint8_t[sizeof(SceneObjectSkyBox::skyboxVertices)];
    std::memcpy(vertexBytes, SceneObjectSkyBox::skyboxVertices.data(),
                sizeof(SceneObjectSkyBox::skyboxVertices));
    SceneObjectVertexArray vertexArray(
        "POSITION", 0, VertexDataType::kVertexDataTypeFloat3, vertexBytes,
        SceneObjectSkyBox::skyboxVertices.size());
    m_dbcSkyBox.position_buffer_offset = CreateVertexBuffer(vertexArray);
    m_dbcSkyBox.property_count = 1;

    auto* indexBytes = new uint8_t[sizeof(indices)];
    std::memcpy(indexBytes, indices.data(), sizeof(indices));
    SceneObjectIndexArray indexArray(
        0, 0, IndexDataType::kIndexDataTypeInt32, indexBytes, indices.size());
    m_dbcSkyBox.index_offset = CreateIndexBuffer(indexArray);
    m_dbcSkyBox.index_count =
        static_cast<uint32_t>(SceneObjectSkyBox::skyboxIndices.size());
    BuildIdentityMatrix(m_dbcSkyBox.modelMatrix);

}

void VulkanGraphicsManager::EndScene() {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    const bool hasResources =
        !m_IndexBuffers.empty() || !m_VertexBuffers.empty() ||
        !m_PerBatchUniformBuffers.empty() || !m_MaterialTextures.empty() ||
        m_DescriptorPool || m_SkyBoxDescriptorPool || m_BRDFDescriptorPool ||
        m_SkyBoxTexture.image || m_BRDFTexture.image ||
        !m_ShadowDescriptors.empty();
    if (hasResources && !app) {
        throw std::logic_error(
            "Vulkan application is unavailable while releasing geometry");
    }

    if (app && hasResources) {
        auto& rhi = app->GetRHI();
        auto device = rhi.GetDevice();
        device.waitIdle();

        if (m_DescriptorPool) {
            device.destroyDescriptorPool(m_DescriptorPool);
            m_DescriptorPool = nullptr;
        }
        if (m_SkyBoxDescriptorPool) {
            device.destroyDescriptorPool(m_SkyBoxDescriptorPool);
            m_SkyBoxDescriptorPool = nullptr;
        }
        if (m_BRDFDescriptorPool) {
            device.destroyDescriptorPool(m_BRDFDescriptorPool);
            m_BRDFDescriptorPool = nullptr;
        }
        m_SkyBoxDescriptorSets.clear();
        m_BRDFDescriptorSet = nullptr;
        for (auto& [pipeline, resources] : m_ShadowDescriptors) {
            if (resources.pool) device.destroyDescriptorPool(resources.pool);
        }
        m_DescriptorSets.clear();
        m_ShadowDescriptors.clear();

        for (auto& buffers : m_PerBatchUniformBuffers) {
            destroyUniformBuffers(buffers);
        }
        m_PerBatchUniformBuffers.clear();

        for (const auto& buffer : m_IndexBuffers) {
            if (buffer.buffer) device.destroyBuffer(buffer.buffer);
            if (buffer.heap) device.freeMemory(buffer.heap);
        }
        for (const auto& buffer : m_VertexBuffers) {
            if (buffer.buffer) device.destroyBuffer(buffer.buffer);
            if (buffer.heap) device.freeMemory(buffer.heap);
        }
        for (const auto& [textureObject, texture] : m_MaterialTextures) {
            if (texture.descriptor) device.destroyImageView(texture.descriptor);
            if (texture.image) device.destroyImage(texture.image);
            if (texture.heap) device.freeMemory(texture.heap);
        }
        m_MaterialTextures.clear();
    }
    m_IndexBuffers.clear();
    m_VertexBuffers.clear();
    m_DrawCommands.clear();
    m_ShadowDrawCommands.clear();
    m_CurrentPipelineState.reset();
    m_CurrentShadowFramebuffer = nullptr;
    m_bRenderingShadowMap = false;
    if (app && m_SkyBoxSampler) {
        app->GetRHI().GetDevice().destroySampler(m_SkyBoxSampler);
        m_SkyBoxSampler = nullptr;
    }

    GraphicsManager::EndScene();
    m_dbcSkyBox = {};
    for (auto& frame : m_Frames) {
        frame.batchContexts.clear();
        frame.skybox = {};
        frame.brdfLUT = {};
    }
}

void VulkanGraphicsManager::BeginFrame(Frame& frame) {
    GraphicsManager::BeginFrame(frame);

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    if (m_bImGuiInitialized) {
        ImGui_ImplVulkan_NewFrame();
    }
    m_nFrameIndex = rhi.GetCurrentFrameIndex();
    assert(frame.frameIndex == m_nFrameIndex);
    m_DrawCommands.clear();
    m_CurrentOffscreenPass.reset();
    SetPerFrameConstants(frame);
    SetLightInfo(frame);
}

void VulkanGraphicsManager::EndFrame(Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    GraphicsManager::EndFrame(frame);
}

void VulkanGraphicsManager::BeginPass(Frame& frame) {
    if (!frame.renderToTexture) return;
    if (!frame.enableMSAA || frame.frameIndex < 0 ||
        frame.colorTextures.size() < 2) {
        throw std::logic_error(
            "Vulkan offscreen scene rendering requires multisampled targets");
    }

    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app || !m_SwapchainResources ||
        !m_SwapchainResources->offscreenRenderPass) {
        throw std::logic_error(
            "Vulkan offscreen render pass is unavailable");
    }
    auto& rhi = app->GetRHI();
    if (static_cast<size_t>(frame.frameIndex) >= m_PerFrameUniformBuffers.size()) {
        throw std::out_of_range("Vulkan offscreen frame index is invalid");
    }
    const auto& resolveColor = frame.colorTextures[0];
    const auto resolveIt = m_RenderTargetTextures.find(resolveColor.handler);
    if (resolveIt == m_RenderTargetTextures.end()) {
        throw std::logic_error(
            "Vulkan offscreen render target resources are unavailable");
    }

    const auto frameIndex = static_cast<uint32_t>(frame.frameIndex);
    const auto samples = rhi.GetMsaaSamples();
    if (m_OffscreenFrames.size() != m_PerFrameUniformBuffers.size() ||
        (!m_OffscreenFrames.empty() &&
         (m_OffscreenFrames.front().width != resolveColor.width ||
          m_OffscreenFrames.front().height != resolveColor.height ||
          m_OffscreenFrames.front().samples != samples))) {
        destroyOffscreenFrames();
        m_OffscreenFrames.resize(m_PerFrameUniformBuffers.size());
    }
    auto& resources = m_OffscreenFrames[frameIndex];
    if (!resources.multisampleColor.image) {
        resources.multisampleColor = rhi.createRenderTexture2D(
            resolveColor.width, resolveColor.height, false, samples);
        try {
            resources.depth = rhi.createRenderTexture2D(
                resolveColor.width, resolveColor.height, true, samples);
        } catch (...) {
            auto device = rhi.GetDevice();
            device.destroyImageView(resources.multisampleColor.descriptor);
            device.destroyImage(resources.multisampleColor.image);
            device.freeMemory(resources.multisampleColor.heap);
            resources = {};
            throw;
        }
        resources.width = resolveColor.width;
        resources.height = resolveColor.height;
        resources.samples = samples;
    }
    if (!resources.framebuffer) {
        resources.framebuffer = rhi.createOffscreenFramebuffer(
            m_SwapchainResources->offscreenRenderPass,
            resources.multisampleColor.descriptor, resources.depth.descriptor,
            resolveIt->second.descriptor, resolveColor.width,
            resolveColor.height);
    }
    m_CurrentOffscreenPass = VulkanRHI::OffscreenPass{
        m_SwapchainResources->offscreenRenderPass, resources.framebuffer,
        vk::Extent2D(resolveColor.width, resolveColor.height),
        {frame.clearColor[0], frame.clearColor[1], frame.clearColor[2], 1.0f}};
}

void VulkanGraphicsManager::EndPass(Frame& frame) {
}

void VulkanGraphicsManager::DrawBatch(const Frame& frame) {
    if (!m_CurrentPipelineState) return;

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    if (frame.frameIndex < 0 ||
        static_cast<size_t>(frame.frameIndex) >=
            m_PerFrameUniformBuffers.size()) {
        throw std::out_of_range("Vulkan frame index exceeds uniform buffers");
    }
    const bool shadowPass = m_CurrentPipelineState->supportsShadowMap;
    if (shadowPass != m_bRenderingShadowMap) {
        throw std::logic_error(
            "Vulkan shadow pipeline and shadow render target are out of sync");
    }

    for (const auto& pDbc : frame.batchContexts) {
        const VulkanDrawBatchContext& dbc =
            dynamic_cast<const VulkanDrawBatchContext&>(*pDbc);

        if (dbc.batchIndex < 0 ||
            static_cast<size_t>(dbc.batchIndex) >=
                m_PerBatchUniformBuffers.size() ||
            dbc.position_buffer_offset >= m_VertexBuffers.size() ||
            dbc.index_offset >= m_IndexBuffers.size()) {
            throw std::out_of_range(
                "Vulkan draw batch references an invalid scene resource");
        }
        const size_t descriptorIndex =
            static_cast<size_t>(dbc.batchIndex) *
                m_PerFrameUniformBuffers.size() +
            frame.frameIndex;
        if (shadowPass) {
            const auto descriptorIt =
                m_ShadowDescriptors.find(m_CurrentPipelineState.get());
            if (descriptorIt == m_ShadowDescriptors.end() ||
                descriptorIndex >= descriptorIt->second.sets.size()) {
                throw std::out_of_range(
                    "Vulkan shadow batch has no descriptor set");
            }
        } else if (dbc.normal_buffer_offset >= m_VertexBuffers.size() ||
                   dbc.texcoord_buffer_offset >= m_VertexBuffers.size() ||
                   dbc.tangent_buffer_offset >= m_VertexBuffers.size() ||
                   dbc.descriptor_offset + frame.frameIndex >=
                       m_DescriptorSets.size()) {
            throw std::out_of_range(
                "Vulkan draw batch references an invalid scene resource");
        }

        const auto* batchConstants =
            static_cast<const PerBatchConstants*>(pDbc.get());
        auto materialParameters = dbc.materialParameters;
        materialParameters[3] = frame.skybox.handler ? 1.0f : 0.0f;
        const VulkanSceneBatchConstants sceneConstants{
            batchConstants->modelMatrix, dbc.baseColor, materialParameters};
        rhi.UpdateUniformBuffer(
            m_PerBatchUniformBuffers[dbc.batchIndex], &sceneConstants,
            sizeof(sceneConstants));

        VulkanRHI::DrawCommand command;
        command.pipeline = m_CurrentPipelineState->pipelineState;
        command.pipelineLayout = m_CurrentPipelineState->pipelineLayout;
        command.vertexBuffers.push_back(
            m_VertexBuffers[dbc.position_buffer_offset].buffer);
        command.vertexOffsets.push_back(0);
        if (!shadowPass) {
            command.vertexBuffers.push_back(
                m_VertexBuffers[dbc.normal_buffer_offset].buffer);
            command.vertexOffsets.push_back(0);
            command.vertexBuffers.push_back(
                m_VertexBuffers[dbc.texcoord_buffer_offset].buffer);
            command.vertexOffsets.push_back(0);
            command.vertexBuffers.push_back(
                m_VertexBuffers[dbc.tangent_buffer_offset].buffer);
            command.vertexOffsets.push_back(0);
        }
        command.indexBuffer = m_IndexBuffers[dbc.index_offset].buffer;
        command.indexType = vk::IndexType::eUint32;
        command.indexCount = dbc.index_count;
        if (shadowPass) {
            command.descriptorSet =
                m_ShadowDescriptors.at(m_CurrentPipelineState.get())
                    .sets[descriptorIndex];
        } else {
            command.descriptorSet =
                m_DescriptorSets[dbc.descriptor_offset + frame.frameIndex];
        }
        if (shadowPass) {
            m_ShadowDrawCommands.push_back(std::move(command));
        } else {
            m_DrawCommands.push_back(std::move(command));
        }
    }
}

void VulkanGraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    m_CurrentPipelineState.reset();
    if (!pipelineState) return;

    auto selectedPipelineState = pipelineState;
    if (pipelineState->pipelineStateName == "PBR") {
        auto* baseApp = dynamic_cast<BaseApplication*>(m_pApp);
        auto* pipelineStateManager =
            baseApp ? dynamic_cast<VulkanPipelineStateManager*>(
                          baseApp->GetPipelineStateManager())
                    : nullptr;
        if (pipelineStateManager) {
            selectedPipelineState =
                pipelineStateManager->GetSceneGeometryPipelineState();
        }
    }

    if (!selectedPipelineState) return;
    auto pState =
        std::dynamic_pointer_cast<VulkanPipelineState>(selectedPipelineState);
    if (!pState ||
        (!pState->supportsSceneGeometry && !pState->supportsSkyBox &&
         !pState->supportsShadowMap && !pState->supportsCompute)) {
        if (m_ReportedUnsupportedPipelineStates
                .insert(selectedPipelineState->pipelineStateName)
                .second) {
            std::cerr << "[VulkanGraphicsManager] Pipeline state '"
                      << selectedPipelineState->pipelineStateName
                      << "' is not implemented; its draw calls will be "
                         "skipped"
                      << std::endl;
        }
        return;
    }
    if (pState->supportsShadowMap &&
        (!m_bRenderingShadowMap || !m_ShadowMapRenderPass)) {
        throw std::logic_error(
            "Vulkan shadow pipeline requested outside a shadow map pass");
    }
    if (pState->supportsSceneGeometry &&
        (!m_SwapchainResources || !m_SwapchainResources->renderPass)) {
        throw std::logic_error(
            "Vulkan scene pipeline requested without an active render pass");
    }
    if (pState->supportsSkyBox &&
        (!m_SwapchainResources || !m_SwapchainResources->renderPass)) {
        throw std::logic_error(
            "Vulkan skybox pipeline requested without an active render pass");
    }
    if (pState->supportsCompute &&
        pState->pipelineType != PIPELINE_TYPE::COMPUTE) {
        throw std::logic_error(
            "Vulkan compute pipeline state has an invalid pipeline type");
    }

    auto device = rhi.GetDevice();
    if (!pState->descriptorSetLayout) {
        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        if (pState->supportsCompute) {
            vk::DescriptorSetLayoutBinding binding;
            binding.binding = 0;
            binding.descriptorType = vk::DescriptorType::eStorageImage;
            binding.descriptorCount = 1;
            binding.stageFlags = vk::ShaderStageFlagBits::eCompute;
            layoutInfo.setBindings(binding);
        } else if (pState->supportsShadowMap) {
            std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
            bindings[0].binding = 11;
            bindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
            bindings[1].binding = 12;
            bindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags =
                vk::ShaderStageFlagBits::eVertex |
                (pState->shadowMapUsesFragmentShader
                     ? vk::ShaderStageFlagBits::eFragment
                     : vk::ShaderStageFlags{});
            bindings[2].binding = 13;
            bindings[2].descriptorType = vk::DescriptorType::eUniformBuffer;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags =
                vk::ShaderStageFlagBits::eVertex |
                (pState->shadowMapUsesFragmentShader
                     ? vk::ShaderStageFlagBits::eFragment
                     : vk::ShaderStageFlags{});
            layoutInfo.setBindings(bindings);
        } else if (pState->supportsSkyBox) {
            std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[1].binding = 5;
            bindings[1].descriptorType = vk::DescriptorType::eSampler;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[2].binding = 10;
            bindings[2].descriptorType = vk::DescriptorType::eUniformBuffer;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags = vk::ShaderStageFlagBits::eVertex |
                                     vk::ShaderStageFlagBits::eFragment;
            layoutInfo.setBindings(bindings);
        } else {
            std::array<vk::DescriptorSetLayoutBinding, 14> bindings{};
            for (uint32_t binding = 0; binding < 5; ++binding) {
                bindings[binding].binding = binding;
                bindings[binding].descriptorType =
                    vk::DescriptorType::eSampledImage;
                bindings[binding].descriptorCount = 1;
                bindings[binding].stageFlags =
                    vk::ShaderStageFlagBits::eFragment;
            }
            bindings[5].binding = 5;
            bindings[5].descriptorType = vk::DescriptorType::eSampler;
            bindings[5].descriptorCount = 1;
            bindings[5].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[6].binding = 6;
            bindings[6].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[6].descriptorCount = 1;
            bindings[6].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[7].binding = 10;
            bindings[7].descriptorType = vk::DescriptorType::eUniformBuffer;
            bindings[7].descriptorCount = 1;
            bindings[7].stageFlags = vk::ShaderStageFlagBits::eVertex |
                                     vk::ShaderStageFlagBits::eFragment;
            bindings[8].binding = 11;
            bindings[8].descriptorType = vk::DescriptorType::eUniformBuffer;
            bindings[8].descriptorCount = 1;
            bindings[8].stageFlags = vk::ShaderStageFlagBits::eVertex |
                                     vk::ShaderStageFlagBits::eFragment;
            bindings[9].binding = 12;
            bindings[9].descriptorType = vk::DescriptorType::eStorageBuffer;
            bindings[9].descriptorCount = 1;
            bindings[9].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[10].binding = 7;
            bindings[10].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[10].descriptorCount = 1;
            bindings[10].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[11].binding = 8;
            bindings[11].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[11].descriptorCount = 1;
            bindings[11].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[12].binding = 9;
            bindings[12].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[12].descriptorCount = 1;
            bindings[12].stageFlags = vk::ShaderStageFlagBits::eFragment;
            bindings[13].binding = 14;
            bindings[13].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[13].descriptorCount = 1;
            bindings[13].stageFlags = vk::ShaderStageFlagBits::eFragment;
            layoutInfo.setBindings(bindings);
        }
        pState->descriptorSetLayout =
            device.createDescriptorSetLayout(layoutInfo);
    }

    if (!pState->pipelineState) {
        if (pState->supportsCompute) {
            rhi.createComputePipeline(
                VulkanRHI::Shader(pState->computeShaderByteCode,
                                  vk::ShaderStageFlagBits::eCompute,
                                  "vulkan_integrateBRDF_comp_main"),
                pState->descriptorSetLayout, pState->pipelineLayout,
                pState->pipelineState);
        } else if (pState->supportsShadowMap) {
            std::array<VulkanRHI::Shader, 2> shaders{
                VulkanRHI::Shader(pState->vertexShaderByteCode,
                                  vk::ShaderStageFlagBits::eVertex,
                                  "vulkan_shadow_vert_main"),
                VulkanRHI::Shader(
                    pState->pixelShaderByteCode,
                    vk::ShaderStageFlagBits::eFragment,
                    "vulkan_shadow_frag_main")};
            vk::VertexInputBindingDescription bindingDescription;
            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(Vector3f);
            bindingDescription.inputRate = vk::VertexInputRate::eVertex;
            vk::VertexInputAttributeDescription attributeDescription;
            attributeDescription.location = 0;
            attributeDescription.binding = 0;
            attributeDescription.format = vk::Format::eR32G32B32Sfloat;
            attributeDescription.offset = 0;
            rhi.createGraphicsPipeline(
                shaders.data(),
                pState->shadowMapUsesFragmentShader ? shaders.size() : 1,
                &bindingDescription, 1, &attributeDescription, 1,
                pState->descriptorSetLayout, m_ShadowMapRenderPass,
                pState->pipelineLayout, pState->pipelineState,
                m_CurrentShadowExtent, true,
                ToVulkanCullMode(pState->cullFaceMode));
        } else if (pState->supportsSkyBox) {
            std::array<VulkanRHI::Shader, 2> shaders{
                VulkanRHI::Shader(pState->vertexShaderByteCode,
                                  vk::ShaderStageFlagBits::eVertex,
                                  "vulkan_skybox_vert_main"),
                VulkanRHI::Shader(pState->pixelShaderByteCode,
                                  vk::ShaderStageFlagBits::eFragment,
                                  "vulkan_skybox_frag_main")};
            vk::VertexInputBindingDescription bindingDescription;
            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(Vector3f);
            bindingDescription.inputRate = vk::VertexInputRate::eVertex;
            vk::VertexInputAttributeDescription attributeDescription;
            attributeDescription.location = 0;
            attributeDescription.binding = 0;
            attributeDescription.format = vk::Format::eR32G32B32Sfloat;
            attributeDescription.offset = 0;
            rhi.createGraphicsPipeline(
                shaders.data(), shaders.size(), &bindingDescription, 1,
                &attributeDescription, 1, pState->descriptorSetLayout,
                m_SwapchainResources->renderPass, pState->pipelineLayout,
                pState->pipelineState, {}, false, vk::CullModeFlagBits::eNone,
                false, vk::CompareOp::eLessOrEqual);
        } else {
        std::array<VulkanRHI::Shader, 2> shaders{
            VulkanRHI::Shader(pState->vertexShaderByteCode,
                              vk::ShaderStageFlagBits::eVertex,
                              "vulkan_scene_vert_main"),
            VulkanRHI::Shader(pState->pixelShaderByteCode,
                              vk::ShaderStageFlagBits::eFragment,
                              "vulkan_scene_frag_main")};
        std::array<vk::VertexInputBindingDescription, 4> bindingDescriptions{};
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vector3f);
        bindingDescriptions[0].inputRate = vk::VertexInputRate::eVertex;
        bindingDescriptions[1].binding = 1;
        bindingDescriptions[1].stride = sizeof(Vector3f);
        bindingDescriptions[1].inputRate = vk::VertexInputRate::eVertex;
        bindingDescriptions[2].binding = 2;
        bindingDescriptions[2].stride = sizeof(Vector2f);
        bindingDescriptions[2].inputRate = vk::VertexInputRate::eVertex;
        bindingDescriptions[3].binding = 3;
        bindingDescriptions[3].stride = sizeof(Vector3f);
        bindingDescriptions[3].inputRate = vk::VertexInputRate::eVertex;
        std::array<vk::VertexInputAttributeDescription, 4>
            attributeDescriptions{};
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].binding = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].binding = 2;
        attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[2].offset = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].binding = 3;
        attributeDescriptions[3].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[3].offset = 0;

        rhi.createGraphicsPipeline(
            shaders.data(), shaders.size(), bindingDescriptions.data(),
            bindingDescriptions.size(), attributeDescriptions.data(),
            attributeDescriptions.size(), pState->descriptorSetLayout,
            m_SwapchainResources->renderPass, pState->pipelineLayout,
            pState->pipelineState);
        }
    }

    if (!pState->supportsCompute) {
        if (pState->supportsShadowMap) {
            createShadowDescriptorSets(*pState, frame);
        } else if (pState->supportsSkyBox) {
            createSkyBoxDescriptorSets(*pState);
        } else {
            createBatchDescriptorSets(*pState, frame);
        }
    }
    m_CurrentPipelineState = std::move(pState);
}

void VulkanGraphicsManager::SetPerFrameConstants(const Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    auto constants =
        *static_cast<const PerFrameConstants*>(&frame.frameContext);
    constants.clip_space_type = rhi.IsSwapChainSrgb() ? 1 : 0;
    rhi.UpdateUniformBuffer(m_PerFrameUniformBuffers, &constants,
                            sizeof(constants));
}

void VulkanGraphicsManager::SetLightInfo(const Frame& frame) {
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    rhi.UpdateUniformBuffer(m_LightStorageBuffers, &frame.lightInfo,
                            sizeof(LightInfo));
}

void VulkanGraphicsManager::Present() {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while presenting");
    }
    if (!m_SwapchainResources || !m_SwapchainResources->renderPass) {
        throw std::logic_error(
            "Vulkan graphics manager has no initialized render pass");
    }
    auto& rhi = app->GetRHI();
    VulkanRHI::RenderCallback renderGui;
    if (m_bImGuiInitialized) {
        renderGui = [](const vk::CommandBuffer& commandBuffer) {
            ImGui_ImplVulkan_RenderDrawData(
                ImGui::GetDrawData(),
                static_cast<VkCommandBuffer>(commandBuffer));
        };
    }
    if (m_CurrentOffscreenPass) {
        rhi.drawFrame(m_SwapchainResources->renderPass, m_DrawCommands,
                      *m_CurrentOffscreenPass, renderGui);
    } else {
        rhi.drawFrame(m_SwapchainResources->renderPass, m_DrawCommands,
                      renderGui);
    }
    m_DrawCommands.clear();
    m_CurrentOffscreenPass.reset();
    m_nFrameIndex = rhi.GetCurrentFrameIndex();
    if (m_PendingCanvasResize) {
        const auto extent = *m_PendingCanvasResize;
        m_PendingCanvasResize.reset();
        GraphicsManager::ResizeCanvas(
            static_cast<int32_t>(extent.width),
            static_cast<int32_t>(extent.height));
    }
}

void VulkanGraphicsManager::ResizeCanvas(int32_t width, int32_t height) {
    if (width > 0 && height > 0) {
        m_PendingCanvasResize =
            vk::Extent2D(static_cast<uint32_t>(width),
                         static_cast<uint32_t>(height));
    }
}

void VulkanGraphicsManager::DrawSkyBox(const Frame& frame) {
    if (!frame.skybox.handler) return;
    if (!m_CurrentPipelineState ||
        !m_CurrentPipelineState->supportsSkyBox) {
        throw std::logic_error(
            "Vulkan skybox draw requested without the skybox pipeline");
    }
    if (frame.frameIndex < 0 ||
        static_cast<size_t>(frame.frameIndex) >=
            m_SkyBoxDescriptorSets.size() ||
        m_dbcSkyBox.position_buffer_offset >= m_VertexBuffers.size() ||
        m_dbcSkyBox.index_offset >= m_IndexBuffers.size()) {
        throw std::out_of_range(
            "Vulkan skybox references an invalid scene resource");
    }

    VulkanRHI::DrawCommand command;
    command.pipeline = m_CurrentPipelineState->pipelineState;
    command.pipelineLayout = m_CurrentPipelineState->pipelineLayout;
    command.vertexBuffers.push_back(
        m_VertexBuffers[m_dbcSkyBox.position_buffer_offset].buffer);
    command.vertexOffsets.push_back(0);
    command.indexBuffer = m_IndexBuffers[m_dbcSkyBox.index_offset].buffer;
    command.indexType = vk::IndexType::eUint32;
    command.indexCount = m_dbcSkyBox.index_count;
    command.descriptorSet =
        m_SkyBoxDescriptorSets[static_cast<size_t>(frame.frameIndex)];
    m_DrawCommands.push_back(std::move(command));
}

void VulkanGraphicsManager::GenerateCubemapArray(
    TextureCubeArray& texture_array) {
    if (texture_array.size == 0 ||
        texture_array.size >
            (std::numeric_limits<uint32_t>::max)() / 6) {
        throw std::invalid_argument("Invalid Vulkan cubemap array size");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    ShadowMapArrayResource resources;
    resources.texture = rhi.createDepthTextureArray(
        texture_array.width, texture_array.height, texture_array.size * 6,
        true, resources.layerViews);
    resources.framebuffers.reserve(resources.layerViews.size());
    try {
        for (const auto view : resources.layerViews) {
            resources.framebuffers.push_back(rhi.createDepthFramebuffer(
                m_ShadowMapRenderPass, view, texture_array.width,
                texture_array.height));
        }
    } catch (...) {
        for (const auto framebuffer : resources.framebuffers) {
            rhi.GetDevice().destroyFramebuffer(framebuffer);
        }
        for (const auto view : resources.layerViews) {
            rhi.GetDevice().destroyImageView(view);
        }
        rhi.GetDevice().destroyImageView(resources.texture.descriptor);
        rhi.GetDevice().destroyImage(resources.texture.image);
        rhi.GetDevice().freeMemory(resources.texture.heap);
        throw;
    }

    texture_array.handler =
        ToTextureHandler(static_cast<VkImageView>(resources.texture.descriptor));
    texture_array.buff =
        ToTextureHandler(static_cast<VkImage>(resources.texture.image));
    const auto [it, inserted] = m_ShadowMapArrays.emplace(
        texture_array.handler, std::move(resources));
    if (!inserted) {
        throw std::logic_error("Duplicate Vulkan depth texture array handle");
    }
}

void VulkanGraphicsManager::GenerateTextureArray(
    Texture2DArray& texture_array) {
    if (texture_array.size == 0) {
        throw std::invalid_argument("Invalid Vulkan depth texture array size");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    ShadowMapArrayResource resources;
    resources.texture = rhi.createDepthTextureArray(
        texture_array.width, texture_array.height, texture_array.size, false,
        resources.layerViews);
    resources.framebuffers.reserve(resources.layerViews.size());
    try {
        for (const auto view : resources.layerViews) {
            resources.framebuffers.push_back(rhi.createDepthFramebuffer(
                m_ShadowMapRenderPass, view, texture_array.width,
                texture_array.height));
        }
    } catch (...) {
        for (const auto framebuffer : resources.framebuffers) {
            rhi.GetDevice().destroyFramebuffer(framebuffer);
        }
        for (const auto view : resources.layerViews) {
            rhi.GetDevice().destroyImageView(view);
        }
        rhi.GetDevice().destroyImageView(resources.texture.descriptor);
        rhi.GetDevice().destroyImage(resources.texture.image);
        rhi.GetDevice().freeMemory(resources.texture.heap);
        throw;
    }

    texture_array.handler =
        ToTextureHandler(static_cast<VkImageView>(resources.texture.descriptor));
    texture_array.buff =
        ToTextureHandler(static_cast<VkImage>(resources.texture.image));
    const auto [it, inserted] = m_ShadowMapArrays.emplace(
        texture_array.handler, std::move(resources));
    if (!inserted) {
        throw std::logic_error("Duplicate Vulkan depth texture array handle");
    }
}

void VulkanGraphicsManager::BeginShadowMap(
    const int32_t light_index, const TextureBase* pShadowmap,
    const int32_t layer_index, const Frame& frame) {
    if (light_index < 0 || light_index >= frame.frameContext.numLights ||
        !pShadowmap || layer_index < 0) {
        throw std::invalid_argument("Invalid Vulkan shadow map request");
    }
    if (m_bRenderingShadowMap) {
        throw std::logic_error("Nested Vulkan shadow map passes are invalid");
    }

    const auto resourceIt = m_ShadowMapArrays.find(pShadowmap->handler);
    const bool cube =
        frame.lightInfo.lights[light_index].lightType == LightType::Omni;
    const size_t baseLayer =
        cube ? static_cast<size_t>(layer_index) * 6
             : static_cast<size_t>(layer_index);
    const size_t layerCount = cube ? 6 : 1;
    if (resourceIt == m_ShadowMapArrays.end() ||
        baseLayer + layerCount > resourceIt->second.framebuffers.size()) {
        throw std::out_of_range("Vulkan shadow map layer is unavailable");
    }
    if (frame.frameIndex < 0 ||
        static_cast<size_t>(frame.frameIndex) >=
            m_ShadowConstantsBuffers.size()) {
        throw std::out_of_range("Vulkan shadow frame index is invalid");
    }

    m_CurrentShadowConstants = {};
    m_CurrentShadowConstants.light_index = light_index;
    m_CurrentShadowConstants.shadowmap_layer_index =
        static_cast<float>(layer_index);
    m_CurrentShadowConstants.near_plane = 1.0f;
    m_CurrentShadowConstants.far_plane = POINT_SHADOW_FAR_PLANE;
    if (cube) {
        const auto& light = frame.lightInfo.lights[light_index];
        const Vector3f position{light.lightPosition[0],
                                light.lightPosition[1],
                                light.lightPosition[2]};
        BuildVulkanPointShadowMatrices(
            m_CurrentShadowConstants.shadowMatrices, position,
            light.lightProjectionMatrix);
    }
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();

    m_CurrentShadowFramebuffer =
        resourceIt->second.framebuffers[baseLayer];
    m_CurrentShadowBaseLayer = baseLayer;
    m_bCurrentShadowCube = cube;
    m_CurrentShadowExtent =
        vk::Extent2D(pShadowmap->width, pShadowmap->height);
    m_ShadowDrawCommands.clear();
    m_bRenderingShadowMap = true;
}

void VulkanGraphicsManager::EndShadowMap(const TextureBase* pShadowmap,
                                        const int32_t layer_index,
                                        const Frame& frame) {
    if (!m_bRenderingShadowMap) return;
    if (!pShadowmap || layer_index < 0 ||
        static_cast<uint32_t>(layer_index) !=
            static_cast<uint32_t>(
                m_CurrentShadowConstants.shadowmap_layer_index) ||
        frame.frameIndex < 0 ||
        static_cast<size_t>(frame.frameIndex) >=
            m_ShadowConstantsBuffers.size()) {
        throw std::invalid_argument("Mismatched Vulkan shadow map end request");
    }
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    auto resetShadowPass = [this] {
        m_ShadowDrawCommands.clear();
        m_CurrentShadowFramebuffer = nullptr;
        m_CurrentShadowExtent = vk::Extent2D{};
        m_CurrentShadowBaseLayer = 0;
        m_bCurrentShadowCube = false;
        m_bRenderingShadowMap = false;
        m_CurrentShadowConstants = {};
    };
    try {
        if (m_bCurrentShadowCube) {
            const auto resourceIt =
                m_ShadowMapArrays.find(pShadowmap->handler);
            if (resourceIt == m_ShadowMapArrays.end()) {
                throw std::logic_error(
                    "Vulkan cube shadow texture was released");
            }
            for (size_t face = 0; face < 6; ++face) {
                m_CurrentShadowConstants.shadowmap_layer_index =
                    static_cast<float>(face);
                rhi.UpdateUniformBuffer(
                    m_ShadowConstantsBuffers[frame.frameIndex],
                    &m_CurrentShadowConstants,
                    sizeof(m_CurrentShadowConstants));
                rhi.drawDepthPass(
                    m_ShadowMapRenderPass,
                    resourceIt->second.framebuffers[m_CurrentShadowBaseLayer +
                                                    face],
                    m_CurrentShadowExtent.width, m_CurrentShadowExtent.height,
                    m_ShadowDrawCommands);
            }
        } else {
            rhi.UpdateUniformBuffer(
                m_ShadowConstantsBuffers[frame.frameIndex],
                &m_CurrentShadowConstants,
                sizeof(m_CurrentShadowConstants));
            rhi.drawDepthPass(m_ShadowMapRenderPass, m_CurrentShadowFramebuffer,
                              m_CurrentShadowExtent.width,
                              m_CurrentShadowExtent.height,
                              m_ShadowDrawCommands);
        }
    } catch (...) {
        resetShadowPass();
        throw;
    }
    resetShadowPass();
    SetLightInfo(frame);
}

void VulkanGraphicsManager::SetShadowMaps(const Frame& frame) {}

void VulkanGraphicsManager::GenerateTexture(Texture2D& texture) {
    if (texture.width == 0 || texture.height == 0 || texture.samples == 0) {
        throw std::invalid_argument("Invalid Vulkan render texture");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    const bool depth = texture.pixel_format == PIXEL_FORMAT::D32;
    const auto samples = texture.samples > 1
                             ? rhi.GetMsaaSamples()
                             : vk::SampleCountFlagBits::e1;
    auto textureResource =
        rhi.createRenderTexture2D(texture.width, texture.height, depth, samples);
    texture.handler = ToTextureHandler(textureResource.descriptor);
    texture.buff = ToTextureHandler(textureResource.image);
    texture.format = static_cast<TextureFormat>(
        static_cast<VkFormat>(textureResource.format));

    bool registered = false;
    try {
        const auto [it, inserted] =
            m_RenderTargetTextures.emplace(texture.handler, textureResource);
        if (!inserted) {
            throw std::logic_error("Duplicate Vulkan render texture view");
        }
        registered = true;
        m_Textures.push_back(texture);
    } catch (...) {
        if (registered) m_RenderTargetTextures.erase(texture.handler);
        auto device = rhi.GetDevice();
        if (textureResource.descriptor) {
            device.destroyImageView(textureResource.descriptor);
        }
        if (textureResource.image) device.destroyImage(textureResource.image);
        if (textureResource.heap) device.freeMemory(textureResource.heap);
        texture = {};
        throw;
    }
}

void VulkanGraphicsManager::CreateTexture(SceneObjectTexture& texture) {}

Texture2D VulkanGraphicsManager::CreateTexture(Image& image) {
    if (!image.data || image.Width == 0 || image.Height == 0 ||
        image.data_size == 0) {
        throw std::invalid_argument("Invalid Vulkan texture image");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    auto textureResource = rhi.createTextureImage(image);
    Texture2D texture;
    texture.handler = ToTextureHandler(textureResource.descriptor);
    texture.buff = ToTextureHandler(textureResource.image);
    texture.format = static_cast<TextureFormat>(
        static_cast<VkFormat>(textureResource.format));
    texture.pixel_format = image.pixel_format;
    texture.width = image.Width;
    texture.height = image.Height;
    texture.mips = 1;

    bool registered = false;
    try {
        const auto [it, inserted] =
            m_StandaloneTextures.emplace(texture.handler, textureResource);
        if (!inserted) {
            throw std::logic_error("Duplicate Vulkan texture image view");
        }
        registered = true;
        m_Textures.push_back(texture);
    } catch (...) {
        if (registered) m_StandaloneTextures.erase(texture.handler);
        auto device = rhi.GetDevice();
        if (textureResource.descriptor) {
            device.destroyImageView(textureResource.descriptor);
        }
        if (textureResource.image) device.destroyImage(textureResource.image);
        if (textureResource.heap) device.freeMemory(textureResource.heap);
        throw;
    }
    return texture;
}

void VulkanGraphicsManager::CreateTextureView(
    Texture2D& textureView, const TextureArrayBase& textureArray,
    uint32_t slice, uint32_t mip) {
    if (!textureArray.handler || mip >= textureArray.mips) {
        throw std::invalid_argument("Invalid Vulkan texture array view");
    }

    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while creating texture view");
    }
    auto& rhi = app->GetRHI();
    TextureViewResource viewResource;
    vk::Format format{};
    if (const auto shadowIt = m_ShadowMapArrays.find(textureArray.handler);
        shadowIt != m_ShadowMapArrays.end()) {
        if (mip != 0 || slice >= shadowIt->second.layerViews.size()) {
            throw std::out_of_range(
                "Vulkan shadow texture view is outside its array");
        }
        viewResource.view = shadowIt->second.layerViews[slice];
        viewResource.source = textureArray.handler;
        format = shadowIt->second.texture.format;
    } else if (textureArray.buff ==
                   ToTextureHandler(m_SkyBoxTexture.image) &&
               m_SkyBoxTexture.image) {
        const uint32_t layerCount = textureArray.size * 6;
        if (slice >= layerCount) {
            throw std::out_of_range(
                "Vulkan skybox texture view is outside its array");
        }
        format = m_SkyBoxTexture.format;
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.image = m_SkyBoxTexture.image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask =
            textureArray.pixel_format == PIXEL_FORMAT::D32
                ? vk::ImageAspectFlagBits::eDepth
                : vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.baseMipLevel = mip;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = slice;
        viewInfo.subresourceRange.layerCount = 1;
        viewResource.view = rhi.GetDevice().createImageView(viewInfo);
        viewResource.source = textureArray.handler;
        viewResource.owned = true;
    } else {
        throw std::invalid_argument(
            "Unsupported Vulkan texture array view source");
    }

    textureView.handler = ToTextureHandler(
        static_cast<VkImageView>(viewResource.view));
    textureView.buff = textureArray.buff;
    textureView.format = static_cast<TextureFormat>(
        static_cast<VkFormat>(format));
    textureView.pixel_format = textureArray.pixel_format;
    textureView.width = (std::max)(1u, textureArray.width >> mip);
    textureView.height = (std::max)(1u, textureArray.height >> mip);
    textureView.mips = 1;

    try {
        const auto [it, inserted] =
            m_TextureViews.emplace(textureView.handler, viewResource);
        if (!inserted) {
            throw std::logic_error("Duplicate Vulkan texture array view");
        }
    } catch (...) {
        if (viewResource.owned) {
            rhi.GetDevice().destroyImageView(viewResource.view);
        }
        textureView = {};
        throw;
    }
}

void VulkanGraphicsManager::UpdateTexture(Texture2D& texture, Image& image) {
    const auto it = m_StandaloneTextures.find(texture.handler);
    if (it == m_StandaloneTextures.end() || texture.width != image.Width ||
        texture.height != image.Height ||
        texture.pixel_format != image.pixel_format || texture.samples != 1) {
        throw std::invalid_argument(
            "Vulkan texture update requires a matching standalone texture");
    }

    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    rhi.updateTextureImage(it->second, image);
}

void VulkanGraphicsManager::ReleaseTexture(TextureBase& texture) {
    const auto textureViewIt = m_TextureViews.find(texture.handler);
    if (textureViewIt != m_TextureViews.end()) {
        auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
        if (!app) {
            throw std::logic_error(
                "Vulkan application is unavailable while releasing texture view");
        }
        auto device = app->GetRHI().GetDevice();
        device.waitIdle();
        removeImGuiTexture(texture.handler);
        if (textureViewIt->second.owned) {
            device.destroyImageView(textureViewIt->second.view);
        }
        m_TextureViews.erase(textureViewIt);
        texture = {};
        return;
    }

    VulkanRHI::Texture* textureResource = nullptr;
    auto standaloneIt = m_StandaloneTextures.find(texture.handler);
    auto renderTargetIt = m_RenderTargetTextures.find(texture.handler);
    if (standaloneIt != m_StandaloneTextures.end()) {
        textureResource = &standaloneIt->second;
    } else if (renderTargetIt != m_RenderTargetTextures.end()) {
        textureResource = &renderTargetIt->second;
    } else if (m_SkyBoxTexture.image &&
        texture.handler == ToTextureHandler(m_SkyBoxTexture.descriptor)) {
        textureResource = &m_SkyBoxTexture;
    } else if (m_BRDFTexture.image &&
               texture.handler == ToTextureHandler(m_BRDFTexture.descriptor)) {
        textureResource = &m_BRDFTexture;
    }
    if (!textureResource) {
        return;
    }
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while releasing texture");
    }
    auto device = app->GetRHI().GetDevice();
    device.waitIdle();
    destroyTextureViewsForSource(texture.handler);
    if (renderTargetIt != m_RenderTargetTextures.end()) {
        destroyOffscreenFrames();
    }
    removeImGuiTexture(texture.handler);
    if (textureResource->descriptor) {
        device.destroyImageView(textureResource->descriptor);
    }
    if (textureResource->image) device.destroyImage(textureResource->image);
    if (textureResource->heap) device.freeMemory(textureResource->heap);
    *textureResource = {};
    if (standaloneIt != m_StandaloneTextures.end()) {
        m_StandaloneTextures.erase(standaloneIt);
    }
    if (renderTargetIt != m_RenderTargetTextures.end()) {
        m_RenderTargetTextures.erase(renderTargetIt);
    }
    texture = {};
}

void VulkanGraphicsManager::GenerateTextureForWrite(Texture2D& texture) {
    if (m_BRDFTexture.image || texture.width == 0 || texture.height == 0 ||
        texture.pixel_format != PIXEL_FORMAT::RG32) {
        throw std::invalid_argument(
            "Vulkan BRDF output requires an unused RG32 texture");
    }
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    m_BRDFTexture = rhi.createStorageTexture2D(
        texture.width, texture.height, vk::Format::eR32G32B32A32Sfloat);
    texture.handler = ToTextureHandler(m_BRDFTexture.descriptor);
    texture.buff = ToTextureHandler(m_BRDFTexture.image);
    texture.format = static_cast<TextureFormat>(
        static_cast<VkFormat>(m_BRDFTexture.format));
    texture.pixel_format = PIXEL_FORMAT::RGBA32;
    texture.mips = 1;
    m_Textures.push_back(texture);
}

void VulkanGraphicsManager::BindTextureForWrite(Texture2D& texture,
                                               const uint32_t slot_index) {
    if (slot_index != 0 || !m_CurrentPipelineState ||
        !m_CurrentPipelineState->supportsCompute || !m_BRDFTexture.image ||
        texture.handler != ToTextureHandler(m_BRDFTexture.descriptor)) {
        throw std::invalid_argument(
            "Vulkan BRDF storage image binding is invalid");
    }
    if (m_BRDFDescriptorPool) {
        throw std::logic_error(
            "Vulkan BRDF descriptor set was already created");
    }
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while binding the BRDF image");
    }
    auto device = app->GetRHI().GetDevice();
    const vk::DescriptorPoolSize poolSize(
        vk::DescriptorType::eStorageImage, 1);
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.maxSets = 1;
    poolInfo.setPoolSizes(poolSize);
    m_BRDFDescriptorPool = device.createDescriptorPool(poolInfo);
    const vk::DescriptorSetLayout layout =
        m_CurrentPipelineState->descriptorSetLayout;
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = m_BRDFDescriptorPool;
    allocInfo.setSetLayouts(layout);
    m_BRDFDescriptorSet = device.allocateDescriptorSets(allocInfo).front();

    vk::DescriptorImageInfo imageInfo(
        nullptr, m_BRDFTexture.descriptor, vk::ImageLayout::eGeneral);
    vk::WriteDescriptorSet write;
    write.dstSet = m_BRDFDescriptorSet;
    write.dstBinding = slot_index;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eStorageImage;
    write.pImageInfo = &imageInfo;
    device.updateDescriptorSets(write, nullptr);
}

void VulkanGraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                    const uint32_t depth) {
    if (!m_CurrentPipelineState ||
        !m_CurrentPipelineState->supportsCompute ||
        !m_BRDFDescriptorSet || !m_BRDFTexture.image ||
        width != m_Frames[0].brdfLUT.width ||
        height != m_Frames[0].brdfLUT.height) {
        throw std::logic_error(
            "Vulkan compute dispatch has no matching BRDF resources");
    }
    auto& rhi = dynamic_cast<VulkanApplication*>(m_pApp)->GetRHI();
    rhi.dispatchCompute(
        m_CurrentPipelineState->pipelineState,
        m_CurrentPipelineState->pipelineLayout, m_BRDFDescriptorSet,
        m_BRDFTexture.image, (width + 7) / 8, (height + 7) / 8, depth);
}

void VulkanGraphicsManager::BeginCompute() {}

void VulkanGraphicsManager::EndCompute() {}

void VulkanGraphicsManager::createBatchDescriptorSets(
    const VulkanPipelineState& pipelineState, const Frame& frame) {
    if (frame.batchContexts.empty() || m_DescriptorPool) return;
    if (m_PerFrameUniformBuffers.empty() ||
        m_PerBatchUniformBuffers.size() < frame.batchContexts.size()) {
        throw std::logic_error(
            "Vulkan scene uniform buffers have not been initialized");
    }

    const size_t frameCount = m_PerFrameUniformBuffers.size();
    if (frameCount == 0 ||
        frame.batchContexts.size() >
            (std::numeric_limits<size_t>::max)() / frameCount) {
        throw std::length_error("Vulkan scene requires too many descriptor sets");
    }
    const size_t setCount = frame.batchContexts.size() * frameCount;
    if (setCount > (std::numeric_limits<uint32_t>::max)() / 10) {
        throw std::length_error("Vulkan scene requires too many descriptor sets");
    }

    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while creating descriptors");
    }
    auto device = app->GetRHI().GetDevice();

    std::array<vk::DescriptorPoolSize, 4> poolSizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                               static_cast<uint32_t>(setCount * 2)),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                               static_cast<uint32_t>(setCount)),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage,
                               static_cast<uint32_t>(setCount * 10)),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler,
                               static_cast<uint32_t>(setCount))};
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.maxSets = static_cast<uint32_t>(setCount);
    poolInfo.setPoolSizes(poolSizes);
    m_DescriptorPool = device.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(
        setCount, pipelineState.descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = m_DescriptorPool;
    allocInfo.setSetLayouts(layouts);
    m_DescriptorSets = device.allocateDescriptorSets(allocInfo);

    auto getShadowView = [this](const TextureBase& texture) {
        const auto it = m_ShadowMapArrays.find(texture.handler);
        if (it == m_ShadowMapArrays.end()) {
            throw std::logic_error(
                "Vulkan scene descriptors require initialized shadow maps");
        }
        return it->second.texture.descriptor;
    };

    for (const auto& pDbc : frame.batchContexts) {
        auto& dbc = dynamic_cast<VulkanDrawBatchContext&>(*pDbc);
        if (dbc.batchIndex < 0 ||
            static_cast<size_t>(dbc.batchIndex) >=
                m_PerBatchUniformBuffers.size()) {
            throw std::out_of_range(
                "Vulkan draw batch has an invalid uniform buffer index");
        }
        dbc.descriptor_offset =
            static_cast<size_t>(dbc.batchIndex) * frameCount;

        for (size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            const size_t descriptorIndex =
                dbc.descriptor_offset + frameIndex;
            vk::DescriptorBufferInfo frameBufferInfo(
                m_PerFrameUniformBuffers[frameIndex].buffer, 0,
                sizeof(PerFrameConstants));
            vk::DescriptorBufferInfo batchBufferInfo(
                m_PerBatchUniformBuffers[dbc.batchIndex][frameIndex].buffer, 0,
                sizeof(VulkanSceneBatchConstants));
            vk::DescriptorBufferInfo lightBufferInfo(
                m_LightStorageBuffers[frameIndex].buffer, 0,
                sizeof(LightInfo));
            std::array<vk::DescriptorImageInfo, 10> imageInfos{};
            for (size_t textureIndex = 0; textureIndex < 5;
                 ++textureIndex) {
                imageInfos[textureIndex].imageView =
                    dbc.materialImageViews[textureIndex];
                imageInfos[textureIndex].imageLayout =
                    vk::ImageLayout::eShaderReadOnlyOptimal;
            }
            imageInfos[5].imageView =
                getShadowView(frame.frameContext.shadowMap);
            imageInfos[5].imageLayout =
                vk::ImageLayout::eDepthStencilReadOnlyOptimal;
            imageInfos[6].imageView =
                getShadowView(frame.frameContext.globalShadowMap);
            imageInfos[6].imageLayout =
                vk::ImageLayout::eDepthStencilReadOnlyOptimal;
            imageInfos[7].imageView =
                getShadowView(frame.frameContext.cubeShadowMap);
            imageInfos[7].imageLayout =
                vk::ImageLayout::eDepthStencilReadOnlyOptimal;
            imageInfos[8].imageView =
                m_SkyBoxTexture.descriptor
                    ? m_SkyBoxTexture.descriptor
                    : m_DefaultEnvironmentTexture.descriptor;
            imageInfos[8].imageLayout =
                vk::ImageLayout::eShaderReadOnlyOptimal;
            imageInfos[9].imageView =
                reinterpret_cast<VkImageView>(frame.brdfLUT.handler);
            imageInfos[9].imageLayout =
                vk::ImageLayout::eShaderReadOnlyOptimal;
            vk::DescriptorImageInfo samplerInfo;
            samplerInfo.sampler = m_MaterialSampler;
            std::array<vk::WriteDescriptorSet, 14> writes{};
            for (uint32_t binding = 0; binding < 5; ++binding) {
                writes[binding].dstSet = m_DescriptorSets[descriptorIndex];
                writes[binding].dstBinding = binding;
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType =
                    vk::DescriptorType::eSampledImage;
                writes[binding].pImageInfo = &imageInfos[binding];
            }
            writes[5].dstSet = m_DescriptorSets[descriptorIndex];
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = vk::DescriptorType::eSampler;
            writes[5].pImageInfo = &samplerInfo;
            writes[6].dstSet = m_DescriptorSets[descriptorIndex];
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = vk::DescriptorType::eSampledImage;
            writes[6].pImageInfo = &imageInfos[8];
            writes[7].dstSet = m_DescriptorSets[descriptorIndex];
            writes[7].dstBinding = 10;
            writes[7].descriptorCount = 1;
            writes[7].descriptorType = vk::DescriptorType::eUniformBuffer;
            writes[7].pBufferInfo = &frameBufferInfo;
            writes[8].dstSet = m_DescriptorSets[descriptorIndex];
            writes[8].dstBinding = 11;
            writes[8].descriptorCount = 1;
            writes[8].descriptorType = vk::DescriptorType::eUniformBuffer;
            writes[8].pBufferInfo = &batchBufferInfo;
            writes[9].dstSet = m_DescriptorSets[descriptorIndex];
            writes[9].dstBinding = 12;
            writes[9].descriptorCount = 1;
            writes[9].descriptorType = vk::DescriptorType::eStorageBuffer;
            writes[9].pBufferInfo = &lightBufferInfo;
            writes[10].dstSet = m_DescriptorSets[descriptorIndex];
            writes[10].dstBinding = 7;
            writes[10].descriptorCount = 1;
            writes[10].descriptorType = vk::DescriptorType::eSampledImage;
            writes[10].pImageInfo = &imageInfos[5];
            writes[11].dstSet = m_DescriptorSets[descriptorIndex];
            writes[11].dstBinding = 8;
            writes[11].descriptorCount = 1;
            writes[11].descriptorType = vk::DescriptorType::eSampledImage;
            writes[11].pImageInfo = &imageInfos[6];
            writes[12].dstSet = m_DescriptorSets[descriptorIndex];
            writes[12].dstBinding = 9;
            writes[12].descriptorCount = 1;
            writes[12].descriptorType = vk::DescriptorType::eSampledImage;
            writes[12].pImageInfo = &imageInfos[7];
            writes[13].dstSet = m_DescriptorSets[descriptorIndex];
            writes[13].dstBinding = 14;
            writes[13].descriptorCount = 1;
            writes[13].descriptorType = vk::DescriptorType::eSampledImage;
            writes[13].pImageInfo = &imageInfos[9];
            device.updateDescriptorSets(writes, nullptr);
        }
    }
}

void VulkanGraphicsManager::createShadowDescriptorSets(
    const VulkanPipelineState& pipelineState, const Frame& frame) {
    auto& resources = m_ShadowDescriptors[&pipelineState];
    if (resources.pool || frame.batchContexts.empty()) return;

    const size_t frameCount = m_ShadowConstantsBuffers.size();
    if (frameCount == 0 ||
        frame.batchContexts.size() >
            (std::numeric_limits<size_t>::max)() / frameCount) {
        throw std::length_error(
            "Vulkan scene requires too many shadow descriptor sets");
    }
    const size_t setCount = frame.batchContexts.size() * frameCount;
    if (setCount > (std::numeric_limits<uint32_t>::max)() / 2) {
        throw std::length_error(
            "Vulkan scene requires too many shadow descriptor sets");
    }

    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while creating shadow "
            "descriptors");
    }
    auto device = app->GetRHI().GetDevice();
    std::array<vk::DescriptorPoolSize, 2> poolSizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                               static_cast<uint32_t>(setCount * 2)),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                               static_cast<uint32_t>(setCount))};
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.maxSets = static_cast<uint32_t>(setCount);
    poolInfo.setPoolSizes(poolSizes);
    resources.pool = device.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(
        setCount, pipelineState.descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = resources.pool;
    allocInfo.setSetLayouts(layouts);
    resources.sets = device.allocateDescriptorSets(allocInfo);

    for (const auto& batchContext : frame.batchContexts) {
        const auto& batch =
            dynamic_cast<const VulkanDrawBatchContext&>(*batchContext);
        for (size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            const size_t descriptorIndex =
                static_cast<size_t>(batch.batchIndex) * frameCount +
                frameIndex;
            vk::DescriptorBufferInfo batchBufferInfo(
                m_PerBatchUniformBuffers[batch.batchIndex][frameIndex].buffer,
                0, sizeof(VulkanSceneBatchConstants));
            vk::DescriptorBufferInfo lightBufferInfo(
                m_LightStorageBuffers[frameIndex].buffer, 0,
                sizeof(LightInfo));
            vk::DescriptorBufferInfo shadowBufferInfo(
                m_ShadowConstantsBuffers[frameIndex].buffer, 0,
                sizeof(ShadowMapConstants));
            std::array<vk::WriteDescriptorSet, 3> writes{};
            const std::array<uint32_t, 3> bindings{11, 12, 13};
            const std::array<vk::DescriptorType, 3> types{
                vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eStorageBuffer,
                vk::DescriptorType::eUniformBuffer};
            const std::array<const vk::DescriptorBufferInfo*, 3> buffers{
                &batchBufferInfo, &lightBufferInfo, &shadowBufferInfo};
            for (size_t i = 0; i < writes.size(); ++i) {
                writes[i].dstSet = resources.sets[descriptorIndex];
                writes[i].dstBinding = bindings[i];
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = types[i];
                writes[i].pBufferInfo = buffers[i];
            }
            device.updateDescriptorSets(writes, nullptr);
        }
    }
}

void VulkanGraphicsManager::createSkyBoxDescriptorSets(
    const VulkanPipelineState& pipelineState) {
    if (m_SkyBoxDescriptorPool || !m_SkyBoxTexture.descriptor) return;
    if (!m_SkyBoxSampler || m_PerFrameUniformBuffers.empty()) {
        throw std::logic_error(
            "Vulkan skybox texture resources are not initialized");
    }

    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        throw std::logic_error(
            "Vulkan application is unavailable while creating skybox "
            "descriptors");
    }
    auto device = app->GetRHI().GetDevice();
    const auto setCount =
        static_cast<uint32_t>(m_PerFrameUniformBuffers.size());
    std::array<vk::DescriptorPoolSize, 3> poolSizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, setCount),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, setCount),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, setCount)};
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.maxSets = setCount;
    poolInfo.setPoolSizes(poolSizes);
    m_SkyBoxDescriptorPool = device.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(
        setCount, pipelineState.descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = m_SkyBoxDescriptorPool;
    allocInfo.setSetLayouts(layouts);
    m_SkyBoxDescriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t frameIndex = 0; frameIndex < setCount; ++frameIndex) {
        vk::DescriptorImageInfo imageInfo(
            nullptr, m_SkyBoxTexture.descriptor,
            vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo samplerInfo(m_SkyBoxSampler, nullptr,
                                            vk::ImageLayout::eUndefined);
        vk::DescriptorBufferInfo bufferInfo(
            m_PerFrameUniformBuffers[frameIndex].buffer, 0,
            sizeof(PerFrameConstants));
        std::array<vk::WriteDescriptorSet, 3> writes{};
        writes[0].dstSet = m_SkyBoxDescriptorSets[frameIndex];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = vk::DescriptorType::eSampledImage;
        writes[0].pImageInfo = &imageInfo;
        writes[1].dstSet = m_SkyBoxDescriptorSets[frameIndex];
        writes[1].dstBinding = 5;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = vk::DescriptorType::eSampler;
        writes[1].pImageInfo = &samplerInfo;
        writes[2].dstSet = m_SkyBoxDescriptorSets[frameIndex];
        writes[2].dstBinding = 10;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = vk::DescriptorType::eUniformBuffer;
        writes[2].pBufferInfo = &bufferInfo;
        device.updateDescriptorSets(writes, nullptr);
    }
}

void VulkanGraphicsManager::destroyShadowMapArrays() {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) return;
    auto device = app->GetRHI().GetDevice();
    for (auto& [handler, resources] : m_ShadowMapArrays) {
        destroyTextureViewsForSource(handler);
        for (const auto framebuffer : resources.framebuffers) {
            if (framebuffer) device.destroyFramebuffer(framebuffer);
        }
        for (const auto view : resources.layerViews) {
            if (view) device.destroyImageView(view);
        }
        if (resources.texture.descriptor) {
            device.destroyImageView(resources.texture.descriptor);
        }
        if (resources.texture.image) device.destroyImage(resources.texture.image);
        if (resources.texture.heap) device.freeMemory(resources.texture.heap);
    }
    m_ShadowMapArrays.clear();
}

void VulkanGraphicsManager::destroyTextureViewsForSource(
    TextureHandler source) {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) return;
    auto device = app->GetRHI().GetDevice();
    for (auto it = m_TextureViews.begin(); it != m_TextureViews.end();) {
        if (it->second.source != source) {
            ++it;
            continue;
        }
        removeImGuiTexture(it->first);
        if (it->second.owned) device.destroyImageView(it->second.view);
        it = m_TextureViews.erase(it);
    }
}

void VulkanGraphicsManager::destroyOffscreenFrames() {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) {
        m_OffscreenFrames.clear();
        return;
    }

    auto& rhi = app->GetRHI();
    auto device = rhi.GetDevice();
    if (device) device.waitIdle();
    for (auto& frame : m_OffscreenFrames) {
        if (frame.framebuffer) device.destroyFramebuffer(frame.framebuffer);
        if (frame.multisampleColor.descriptor) {
            device.destroyImageView(frame.multisampleColor.descriptor);
        }
        if (frame.multisampleColor.image) {
            device.destroyImage(frame.multisampleColor.image);
        }
        if (frame.multisampleColor.heap) {
            device.freeMemory(frame.multisampleColor.heap);
        }
        if (frame.depth.descriptor) device.destroyImageView(frame.depth.descriptor);
        if (frame.depth.image) device.destroyImage(frame.depth.image);
        if (frame.depth.heap) device.freeMemory(frame.depth.heap);
    }
    m_OffscreenFrames.clear();
}

void VulkanGraphicsManager::destroyUniformBuffers(
    std::vector<VulkanRHI::UniformBuffer>& uniformBuffers) {
    auto* app = dynamic_cast<VulkanApplication*>(m_pApp);
    if (!app) return;
    auto device = app->GetRHI().GetDevice();
    for (const auto& buffer : uniformBuffers) {
        if (buffer.buffer) device.destroyBuffer(buffer.buffer);
        if (buffer.heap) device.freeMemory(buffer.heap);
    }
    uniformBuffers.clear();
}

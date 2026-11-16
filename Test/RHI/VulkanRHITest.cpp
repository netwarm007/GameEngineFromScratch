#include "AssetLoader.hpp"
#include "PVR.hpp"
#include "SceneObjectIndexArray.hpp"
#include "SceneObjectVertexArray.hpp"
#include "Vulkan/VulkanGraphicsManager.hpp"
#include "Vulkan/VulkanRHI.hpp"
#include "VulkanApplication.hpp"
#include "config.h"
#include "imgui.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace My;

class TestVulkanApplication : public VulkanApplication {
   public:
    using VulkanApplication::VulkanApplication;

    void SetPipelineStateManagerForTest(IPipelineStateManager* manager) {
        m_pPipelineStateManager = manager;
    }
};

class TestVulkanGraphicsManager : public VulkanGraphicsManager {
   public:
    using VulkanGraphicsManager::CreateIndexBuffer;
    using VulkanGraphicsManager::CreateVertexBuffer;
    using VulkanGraphicsManager::BeginCompute;
    using VulkanGraphicsManager::BeginPass;
    using VulkanGraphicsManager::EndCompute;
    using VulkanGraphicsManager::EndPass;
    using VulkanGraphicsManager::EndScene;
    using VulkanGraphicsManager::PendingDrawCommandCount;

    void BeginSceneForTest(const Scene& scene) { BeginScene(scene); }
    void BeginFrameForTest(Frame& frame) { BeginFrame(frame); }
    Frame& GetTestFrame() { return m_Frames[0]; }
    Frame& GetTestFrame(size_t index) { return m_Frames.at(index); }
    void RunBRDFForTest(const std::shared_ptr<PipelineState>& pipelineState) {
        auto& frame = GetTestFrame();
        Texture2D brdfLUT;
        brdfLUT.width = 8;
        brdfLUT.height = 8;
        brdfLUT.pixel_format = PIXEL_FORMAT::RG32;
        BeginCompute();
        SetPipelineState(pipelineState, frame);
        GenerateTextureForWrite(brdfLUT);
        frame.brdfLUT = brdfLUT;
        BindTextureForWrite(brdfLUT, 0);
        Dispatch(brdfLUT.width, brdfLUT.height, 1);
        EndCompute();
    }
    Vector4f GetTestBatchBaseColor() const {
        return static_cast<const VulkanDrawBatchContext&>(
                   *m_Frames[0].batchContexts[0])
            .baseColor;
    }
};

struct Vertex {
    Vector3f pos;
    Vector2f texCoord;

    static constexpr vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription;
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, 2>
    getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2>
            attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    Matrix4X4f model;
    Matrix4X4f view;
    Matrix4X4f proj;
};

int main(int argc, char** argv) {
    try {
        uint32_t frameLimit = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            constexpr std::string_view frameOption = "--frames=";
            if (argument.starts_with(frameOption)) {
                const auto value = argument.substr(frameOption.size());
                size_t parsedCharacters = 0;
                const auto frameCount =
                    std::stoul(value, &parsedCharacters);
                if (frameCount == 0 ||
                    parsedCharacters != value.size() ||
                    frameCount >
                        (std::numeric_limits<uint32_t>::max)()) {
                    throw std::invalid_argument(
                        "--frames must be between 1 and UINT32_MAX");
                }
                frameLimit = static_cast<uint32_t>(frameCount);
            } else {
                throw std::invalid_argument("unknown argument: " + argument);
            }
        }

        GfxConfiguration config(8, 8, 8, 8, 24, 8, 4, 800, 600,
                                "Vulkan RHI Test");
        TestVulkanApplication app(config);

        AssetLoader asset_loader;

        assert(asset_loader.Initialize() == 0 &&
               "Asset Loader Initialize Failed!");

        // 创建窗口
        {
            assert(app.Initialize() == 0 && "App Initialize Failed!");

            app.CreateMainWindow();
        }

        auto& rhi = app.GetRHI();

        // 获取命令队列
        rhi.getDeviceQueues();

        VulkanPipelineStateManager pipelineStateManager;
        pipelineStateManager.SetAppPointer(&app);
        if (pipelineStateManager.Initialize() != 0) {
            throw std::runtime_error(
                "Vulkan pipeline state manager failed to initialize");
        }
        app.SetPipelineStateManagerForTest(&pipelineStateManager);

        TestVulkanGraphicsManager graphicsManager;
        graphicsManager.SetAppPointer(&app);
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        if (graphicsManager.Initialize() != 0) {
            throw std::runtime_error(
                "Vulkan graphics manager failed to initialize");
        }
        graphicsManager.RunBRDFForTest(
            pipelineStateManager.GetPipelineState("PBR BRDF CS"));

        Image standaloneTextureImage;
        standaloneTextureImage.Width = 1;
        standaloneTextureImage.Height = 1;
        standaloneTextureImage.bitcount = 32;
        standaloneTextureImage.bitdepth = 8;
        standaloneTextureImage.pitch = 4;
        standaloneTextureImage.data_size = 4;
        standaloneTextureImage.pixel_format = PIXEL_FORMAT::RGBA8;
        standaloneTextureImage.data =
            new uint8_t[4]{255, 255, 255, 255};
        auto standaloneTexture =
            graphicsManager.CreateTexture(standaloneTextureImage);
        delete[] standaloneTextureImage.data;
        standaloneTextureImage.data = nullptr;
        if (!standaloneTexture.handler || !standaloneTexture.buff ||
            standaloneTexture.width != 1 || standaloneTexture.height != 1) {
            throw std::runtime_error(
                "Vulkan graphics manager failed to create a standalone texture");
        }
        standaloneTextureImage.data =
            new uint8_t[4]{0, 0, 0, 255};
        graphicsManager.UpdateTexture(standaloneTexture,
                                      standaloneTextureImage);
        delete[] standaloneTextureImage.data;
        standaloneTextureImage.data = nullptr;
        graphicsManager.ReleaseTexture(standaloneTexture);
        if (standaloneTexture.handler || standaloneTexture.buff) {
            throw std::runtime_error(
                "Vulkan graphics manager failed to release a standalone texture");
        }

        const std::array<float, 9> managerVertices{};
        auto* managerVertexBytes = new uint8_t[sizeof(managerVertices)];
        std::memcpy(managerVertexBytes, managerVertices.data(),
                    sizeof(managerVertices));
        SceneObjectVertexArray managerVertexArray(
            "POSITION", 0, VertexDataType::kVertexDataTypeFloat3,
            managerVertexBytes, managerVertices.size());
        if (graphicsManager.CreateVertexBuffer(managerVertexArray) != 0) {
            throw std::runtime_error(
                "Vulkan graphics manager returned an invalid vertex buffer "
                "index");
        }

        const std::array<uint32_t, 3> managerIndices{0, 1, 2};
        auto* managerIndexBytes = new uint8_t[sizeof(managerIndices)];
        std::memcpy(managerIndexBytes, managerIndices.data(),
                    sizeof(managerIndices));
        SceneObjectIndexArray managerIndexArray(
            0, 0, IndexDataType::kIndexDataTypeInt32, managerIndexBytes,
            managerIndices.size());
        if (graphicsManager.CreateIndexBuffer(managerIndexArray) != 0) {
            throw std::runtime_error(
                "Vulkan graphics manager returned an invalid index buffer "
                "index");
        }

        const std::array<uint16_t, 3> compactIndices{0, 1, 2};
        auto* compactIndexBytes = new uint8_t[sizeof(compactIndices)];
        std::memcpy(compactIndexBytes, compactIndices.data(),
                    sizeof(compactIndices));
        SceneObjectIndexArray compactIndexArray(
            0, 0, IndexDataType::kIndexDataTypeInt16, compactIndexBytes,
            compactIndices.size());
        if (graphicsManager.CreateIndexBuffer(compactIndexArray) != 1) {
            throw std::runtime_error(
                "Vulkan graphics manager failed to accept 16-bit indices");
        }

        auto createSkyboxImage = [](uint32_t width, bool fullMipChain,
                                    bool rgb = false) {
            auto image = std::make_shared<Image>();
            image->Width = width;
            image->Height = width;
            const size_t channels = rgb ? 3 : 4;
            image->bitcount = static_cast<uint16_t>(channels * 8);
            image->bitdepth = 8;
            image->pixel_format =
                rgb ? PIXEL_FORMAT::RGB8 : PIXEL_FORMAT::RGBA8;
            uint32_t mipWidth = width;
            size_t offset = 0;
            do {
                const size_t pitch = static_cast<size_t>(mipWidth) * channels;
                const size_t size = pitch * mipWidth;
                image->mipmaps.emplace_back(mipWidth, mipWidth, pitch, offset,
                                            size);
                offset += size;
                mipWidth = std::max(1u, mipWidth >> 1);
            } while (fullMipChain && mipWidth < image->mipmaps.back().Width);
            image->data_size = offset;
            image->pitch = static_cast<size_t>(width) * channels;
            image->data = new uint8_t[offset];
            std::memset(image->data, 128, offset);
            if (!rgb) {
                for (uint32_t mip = 0; mip < image->mipmaps.size(); ++mip) {
                    const auto& mipmap = image->mipmaps[mip];
                    for (uint32_t pixel = 0;
                         pixel < mipmap.Width * mipmap.Height; ++pixel) {
                        image->data[mipmap.offset + pixel * 4 + 3] = 255;
                    }
                }
            }
            return image;
        };

        Scene scene("Vulkan geometry test");
        for (uint32_t face = 0; face < 6; ++face) {
            scene.SkyBox->GetTexture(face).SetTextureImage(
                createSkyboxImage(2, false));
            scene.SkyBox->GetTexture(face + 6).SetTextureImage(
                createSkyboxImage(1, false));
            scene.SkyBox->GetTexture(face + 12).SetTextureImage(
                createSkyboxImage(2, true, face == 0));
        }
        auto material = std::make_shared<SceneObjectMaterial>("test material");
        const std::string diffuseAttribute = "diffuse";
        const Vector4f expectedBaseColor{0.25f, 0.5f, 0.75f, 1.0f};
        material->SetColor(diffuseAttribute, expectedBaseColor);
        auto baseColorTexture = std::make_shared<SceneObjectTexture>(
            "Vulkan base-color texture");
        auto baseColorImage = std::make_shared<Image>();
        baseColorImage->Width = 1;
        baseColorImage->Height = 1;
        baseColorImage->bitcount = 24;
        baseColorImage->bitdepth = 8;
        baseColorImage->pitch = 4;
        baseColorImage->data_size = 4;
        baseColorImage->pixel_format = PIXEL_FORMAT::RGB8;
        baseColorImage->data = new uint8_t[4]{255, 255, 255, 0};
        baseColorTexture->SetTextureImage(std::move(baseColorImage));
        material->SetTexture(diffuseAttribute, baseColorTexture);
        auto normalTexture = std::make_shared<SceneObjectTexture>(
            "Vulkan normal texture");
        auto normalImage = std::make_shared<Image>();
        normalImage->Width = 1;
        normalImage->Height = 1;
        normalImage->bitcount = 32;
        normalImage->bitdepth = 8;
        normalImage->pitch = 8;
        normalImage->data_size = 8;
        normalImage->pixel_format = PIXEL_FORMAT::RGBA8;
        normalImage->data =
            new uint8_t[8]{128, 128, 255, 255, 0, 0, 0, 0};
        normalTexture->SetTextureImage(std::move(normalImage));
        material->SetTexture("normal", normalTexture);
        scene.Materials.emplace("test material", material);
        auto geometry = std::make_shared<SceneObjectGeometry>();
        auto mesh = std::make_shared<SceneObjectMesh>();
        const std::array<float, 9> triangleVertices{
            -0.6f, -0.6f, 0.5f, 0.6f, -0.6f, 0.5f, 0.0f, 0.6f, 0.5f};
        auto* triangleVertexBytes = new uint8_t[sizeof(triangleVertices)];
        std::memcpy(triangleVertexBytes, triangleVertices.data(),
                    sizeof(triangleVertices));
        mesh->AddVertexArray(SceneObjectVertexArray(
            "position", 0, VertexDataType::kVertexDataTypeFloat3,
            triangleVertexBytes, triangleVertices.size()));
        const std::array<float, 6> triangleTexcoords{
            0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
        auto* triangleTexcoordBytes =
            new uint8_t[sizeof(triangleTexcoords)];
        std::memcpy(triangleTexcoordBytes, triangleTexcoords.data(),
                    sizeof(triangleTexcoords));
        mesh->AddVertexArray(SceneObjectVertexArray(
            "texcoord", 0, VertexDataType::kVertexDataTypeFloat2,
            triangleTexcoordBytes, triangleTexcoords.size()));
        const std::array<uint16_t, 3> triangleIndices{0, 1, 2};
        auto* triangleIndexBytes = new uint8_t[sizeof(triangleIndices)];
        std::memcpy(triangleIndexBytes, triangleIndices.data(),
                    sizeof(triangleIndices));
        mesh->AddIndexArray(SceneObjectIndexArray(
            0, 0, IndexDataType::kIndexDataTypeInt16, triangleIndexBytes,
            triangleIndices.size()));
        geometry->AddMesh(std::move(mesh));
        scene.Geometries.emplace("triangle", geometry);
        auto geometryNode =
            std::make_shared<SceneGeometryNode>("triangle node");
        geometryNode->AddSceneObjectRef("triangle");
        geometryNode->AddMaterialRef("test material");
        geometryNode->SetVisibility(true);
        scene.GeometryNodes.emplace("triangle", geometryNode);

        graphicsManager.BeginSceneForTest(scene);
        const auto actualBaseColor =
            graphicsManager.GetTestBatchBaseColor();
        for (size_t component = 0; component < 4; ++component) {
            if (actualBaseColor[component] != expectedBaseColor[component]) {
                throw std::runtime_error(
                    "Vulkan scene batch did not preserve material base color");
            }
        }
        BuildIdentityMatrix(
            graphicsManager.GetTestFrame().batchContexts[0]->modelMatrix);
        auto configureFrame = [](Frame& frame) {
            BuildIdentityMatrix(frame.frameContext.viewMatrix);
            BuildIdentityMatrix(frame.frameContext.projectionMatrix);
            frame.frameContext.camPos = {0.0f, 0.0f, 2.0f, 1.0f};
            frame.frameContext.numLights = 1;
            auto& light = frame.lightInfo.lights[0];
            light.lightColor = {1.0f, 1.0f, 1.0f, 1.0f};
            light.lightIntensity = 10.0f;
            light.lightCastShadow = 1;
            light.lightDirection = {0.0f, 0.0f, -1.0f, 0.0f};
            BuildIdentityMatrix(light.lightViewMatrix);
            BuildIdentityMatrix(light.lightProjectionMatrix);
        };
        auto& sceneFrame = graphicsManager.GetTestFrame();
        configureFrame(sceneFrame);
        graphicsManager.BeginFrameForTest(sceneFrame);
        sceneFrame.renderToTexture = true;
        graphicsManager.BeginPass(sceneFrame);
        Texture2D shadowTextureView;
        graphicsManager.CreateTextureView(
            shadowTextureView, sceneFrame.frameContext.shadowMap, 0, 0);
        Texture2D skyboxTextureView;
        graphicsManager.CreateTextureView(skyboxTextureView, sceneFrame.skybox,
                                          0, 0);
        ImGui::GetIO().DisplaySize = ImVec2(800.0f, 600.0f);
        ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
        ImGui::NewFrame();
        ImGui::Begin("Vulkan ImGui backend test");
        ImGui::Text("Vulkan scene rendering");
        ImGui::Image(
            (ImTextureID)graphicsManager.GetImGuiTextureHandle(
                sceneFrame.colorTextures[0]),
            ImVec2(128.0f, 128.0f));
        ImGui::Image(
            (ImTextureID)graphicsManager.GetImGuiTextureHandle(
                shadowTextureView),
            ImVec2(64.0f, 64.0f));
        ImGui::Image(
            (ImTextureID)graphicsManager.GetImGuiTextureHandle(
                skyboxTextureView),
            ImVec2(64.0f, 64.0f));
        ImGui::End();
        ImGui::Render();

        auto renderShadowMap =
            [&](LightType lightType, TextureBase& shadowMap,
                const char* shadowPipelineName) {
                sceneFrame.lightInfo.lights[0].lightType = lightType;
                sceneFrame.lightInfo.lights[0].lightPosition =
                    {0.0f, 0.0f, 2.0f,
                     lightType == LightType::Infinity ? 0.0f : 1.0f};
                graphicsManager.BeginShadowMap(0, &shadowMap, 0, sceneFrame);
                graphicsManager.SetPipelineState(
                    pipelineStateManager.GetPipelineState(shadowPipelineName),
                    sceneFrame);
                graphicsManager.DrawBatch(sceneFrame);
                graphicsManager.EndShadowMap(&shadowMap, 0, sceneFrame);
            };

        renderShadowMap(LightType::Omni,
                        sceneFrame.frameContext.cubeShadowMap,
                        "Omni Light Shadow Map");
        renderShadowMap(LightType::Spot, sceneFrame.frameContext.shadowMap,
                        "Spot Light Shadow Map");
        renderShadowMap(LightType::Infinity,
                        sceneFrame.frameContext.globalShadowMap,
                        "Sun Light Shadow Map");
        graphicsManager.SetPipelineState(
            pipelineStateManager.GetPipelineState("PBR"), sceneFrame);
        graphicsManager.DrawBatch(sceneFrame);
        graphicsManager.SetPipelineState(
            pipelineStateManager.GetPipelineState("SkyBox"), sceneFrame);
        graphicsManager.DrawSkyBox(sceneFrame);
        graphicsManager.EndPass(sceneFrame);
        if (graphicsManager.PendingDrawCommandCount() != 2) {
            throw std::runtime_error(
                "Vulkan engine scene did not produce geometry and skybox draws");
        }
        graphicsManager.Present();
        graphicsManager.ReleaseTexture(shadowTextureView);
        graphicsManager.ReleaseTexture(skyboxTextureView);

        auto renderOffscreenFrame = [&](Frame& frame, const char* windowTitle,
                                        bool resizeCanvas) {
            configureFrame(frame);
            graphicsManager.BeginFrameForTest(frame);
            frame.renderToTexture = true;
            graphicsManager.BeginPass(frame);
            ImGui::NewFrame();
            ImGui::Begin(windowTitle);
            ImGui::Image(
                (ImTextureID)graphicsManager.GetImGuiTextureHandle(
                    frame.colorTextures[0]),
                ImVec2(64.0f, 64.0f));
            ImGui::End();
            ImGui::Render();
            graphicsManager.SetPipelineState(
                pipelineStateManager.GetPipelineState("PBR"), frame);
            graphicsManager.DrawBatch(frame);
            graphicsManager.SetPipelineState(
                pipelineStateManager.GetPipelineState("SkyBox"), frame);
            graphicsManager.DrawSkyBox(frame);
            graphicsManager.EndPass(frame);
            if (resizeCanvas) graphicsManager.ResizeCanvas(640, 480);
            graphicsManager.Present();
        };
        renderOffscreenFrame(graphicsManager.GetTestFrame(1),
                             "Vulkan offscreen frame reuse test", false);
        renderOffscreenFrame(graphicsManager.GetTestFrame(0),
                             "Vulkan offscreen resize test", true);
        if (graphicsManager.GetTestFrame(0).colorTextures[0].width != 640 ||
            graphicsManager.GetTestFrame(0).colorTextures[0].height != 480) {
            throw std::runtime_error(
                "Vulkan offscreen resize was not applied after presentation");
        }
        renderOffscreenFrame(graphicsManager.GetTestFrame(1),
                             "Vulkan resized offscreen test", false);

        rhi.GetDevice().waitIdle();
        pipelineStateManager.InvalidatePipelines();
        graphicsManager.Finalize();
        pipelineStateManager.Finalize();
        ImGui::DestroyContext();

        std::vector<VulkanRHI::Shader> vkShaders;

        vk::RenderPass vkRenderPass;
        vk::Pipeline vkGraphicPipeline;
        vk::DescriptorSetLayout vkDescriptorSetLayout;
        vk::PipelineLayout vkPipelineLayout;
        vk::Sampler vkTextureSampler;

        std::vector<VulkanRHI::Texture> textures;
        std::vector<VulkanRHI::IndexBuffer> indexBuffers;
        std::vector<VulkanRHI::VertexBuffer> vertexBuffers;
        std::vector<VulkanRHI::UniformBuffer> uniformBuffers;

        // 创建 Descriptor 布局
        vkDescriptorSetLayout = rhi.createDescriptorSetLayout();

        // 创建图形管道
        {
            auto vertShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/simple.vert.spv");
            auto fragShader = asset_loader.SyncOpenAndReadBinary(
                "Shaders/Vulkan/simple.frag.spv");
            vkShaders.emplace_back(rhi.createShaderModule(vertShader), vk::ShaderStageFlagBits::eVertex, "simple_vert_main");
            vkShaders.emplace_back(rhi.createShaderModule(fragShader), vk::ShaderStageFlagBits::eFragment, "simple_frag_main");
        }

        // 加载贴图
        {
            Image image;
            if (frameLimit > 0) {
                image.Width = 1;
                image.Height = 1;
                image.bitcount = 32;
                image.data_size = 4;
                image.data = new uint8_t[4]{255, 255, 255, 255};
            } else {
                auto buf = asset_loader.SyncOpenAndReadBinary(
                    "Textures/viking_room.pvr");
                PVR::PvrParser parser;
                image = parser.Parse(buf);
            }
            textures.emplace_back(rhi.createTextureImage(image));
        }

        // 创建采样器
        vkTextureSampler = rhi.createTextureSampler();

        // 加载模型
        {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            auto model_path =
                asset_loader.GetFileRealPath("Models/viking_room.obj");

            assert(!model_path.empty() && "Can not find model file!");

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn,
                                    &err, model_path.c_str())) {
                throw std::runtime_error(warn + err);
            }

            for (const auto& shape : shapes) {
                for (const auto& index : shape.mesh.indices) {
                    Vertex vertex{};

                    vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]};

                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f -
                            attrib.texcoords[2 * index.texcoord_index +
                                                1]};

                    vertices.push_back(vertex);
                    indices.push_back(indices.size());
                }
            }

            // 创建顶点缓冲区
            vertexBuffers.resize(1);
            rhi.createVertexBuffer(vertices.data(), sizeof(vertices[0]) * vertices.size(), vertexBuffers[0]);

            // 创建索引缓冲区
            indexBuffers.resize(1);
            rhi.createIndexBuffer(indices.data(), sizeof(indices[0]) * indices.size(), indexBuffers[0]);
        }


        // 创建常量缓冲区
        rhi.createUniformBuffers(uniformBuffers, sizeof(UniformBufferObject));

        // 创建资源描述子池
        rhi.createDescriptorPool();

        // 创建资源描述子集
        rhi.createDescriptorSets(vkDescriptorSetLayout, uniformBuffers, textures, vkTextureSampler);

        VulkanRHI::CreateSwapChainCBFunc createSwapChainCBFunc =
            [&rhi, &vkShaders, &vkGraphicPipeline, &vkPipelineLayout,
             &vkRenderPass, &vkDescriptorSetLayout]() {
                // 创建渲染工序（Render Pass）
                vkRenderPass = rhi.createRenderPass();

                // 创建图形管道
                {
                    auto bindingDesc = Vertex::getBindingDescription();
                    auto attrDescs = Vertex::getAttributeDescriptions();
                    rhi.createGraphicsPipeline(
                        vkShaders.data(), vkShaders.size(), bindingDesc,
                        attrDescs.data(), attrDescs.size(),
                        vkDescriptorSetLayout, vkRenderPass, vkPipelineLayout,
                        vkGraphicPipeline);
                }

                // 创建backend RT
                rhi.createColorResources();

                // 创建深度缓冲区
                rhi.createDepthResources();

                // 创建 Framebuffers
                rhi.createFramebuffers(vkRenderPass);

            };

        rhi.CreateSwapChainCB(createSwapChainCBFunc);

        // 创建并登记资源销毁回调函数
        VulkanRHI::DestroySwapChainCBFunc destroySwapChainCBFunc =
            [&rhi, &vkGraphicPipeline, &vkPipelineLayout,
             &vkRenderPass]() {
                auto device = rhi.GetDevice();

                // 释放图形管道
                device.destroyPipeline(vkGraphicPipeline);

                // 释放管道布局
                device.destroyPipelineLayout(vkPipelineLayout);

                // 释放Render Pass
                device.destroyRenderPass(vkRenderPass);
            };

        rhi.DestroySwapChainCB(destroySwapChainCBFunc);

        // 创建图形渲染交换链
        rhi.CreateSwapChain();

        std::vector<VulkanRHI::DrawCommand> drawCommands(2);
        for (auto& draw : drawCommands) {
            draw.pipeline = vkGraphicPipeline;
            draw.pipelineLayout = vkPipelineLayout;
            draw.vertexBuffers.push_back(vertexBuffers[0].buffer);
            draw.vertexOffsets.push_back(0);
            draw.indexBuffer = indexBuffers[0].buffer;
            draw.indexCount = indexBuffers[0].indexCount;
        }

        // 主消息循环
        uint32_t renderedFrames = 0;
        while (!app.IsQuit() &&
               (frameLimit == 0 || renderedFrames < frameLimit)) {
            app.Tick();

            static auto startTime = std::chrono::high_resolution_clock::now();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(
                            currentTime - startTime)
                            .count();

            UniformBufferObject ubo{};
            BuildIdentityMatrix(ubo.model);
            MatrixRotationAxis(ubo.model, {0.0f, 0.0f, 1.0f}, time * PI / 8.0f);
            BuildViewRHMatrix(ubo.view, {2.0f, 2.0f, 2.0f}, {0.0f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 1.0f});
            uint32_t width, height;
            app.GetFramebufferSize(width, height);
            BuildPerspectiveFovRHMatrix(ubo.proj, PI / 4.0f, width / (float)height,
                                        0.1f, 10.0f);
            ubo.proj[1][1] *= -1.0f;

            rhi.UpdateUniformBuffer(uniformBuffers, &ubo, sizeof(ubo));

            if (renderedFrames == 0) {
                rhi.drawFrame(vkRenderPass, vkGraphicPipeline,
                              vkPipelineLayout, vertexBuffers[0].buffer,
                              indexBuffers[0].buffer,
                              indexBuffers[0].indexCount);
            } else {
                rhi.drawFrame(vkRenderPass, drawCommands);
            }
            ++renderedFrames;
        }

        auto device = rhi.GetDevice();

        // 释放着色器资源描述子布局
        device.destroyDescriptorSetLayout(vkDescriptorSetLayout);

        // 释放着色器模块
        std::for_each(vkShaders.begin(), vkShaders.end(),
                        [&device](auto& shader) {
                            device.destroyShaderModule(shader.module);
                        });
        vkShaders.clear();

        // 释放贴图采样器
        device.destroySampler(vkTextureSampler);

        // 释放贴图
        std::for_each(textures.begin(), textures.end(),
                        [&device](VulkanRHI::Texture& tex) {
                            device.destroyImageView(tex.descriptor);
                            device.destroyImage(tex.image);
                            device.freeMemory(tex.heap);
                        });
        textures.clear();

        // 释放Uniform Buffer
        std::for_each(uniformBuffers.begin(), uniformBuffers.end(),
                        [&device](VulkanRHI::UniformBuffer& ubo) {
                            device.destroyBuffer(ubo.buffer);
                            device.freeMemory(ubo.heap);
                        });
        uniformBuffers.clear();

        // 释放索引缓冲区
        std::for_each(indexBuffers.begin(), indexBuffers.end(),
                        [&device](auto& idxBuf) {
                            device.destroyBuffer(idxBuf.buffer);
                            device.freeMemory(idxBuf.heap);
                        });
        indexBuffers.clear();

        // 释放顶点缓冲区
        std::for_each(vertexBuffers.begin(), vertexBuffers.end(),
                        [&device](auto& vertBuf) {
                            device.destroyBuffer(vertBuf.buffer);
                            device.freeMemory(vertBuf.heap);
                        });
        vertexBuffers.clear();

        app.Finalize();
    } catch (vk::SystemError& err) {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return (-1);
    } catch (std::exception& err) {
        std::cout << "std::exception: " << err.what() << std::endl;
        return (-1);
    } catch (...) {
        std::cout << "unknown error\n";
        return (-1);
    }

    return 0;
}
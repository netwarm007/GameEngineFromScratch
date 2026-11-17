#include "Vulkan/VulkanGraphicsManager.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "AssetLoader.hpp"
#include "BRDFIntegrator.hpp"
#include "BrdfReference.hpp"
#include "ISceneManager.hpp"
#include "GeometrySubPass.hpp"
#include "ForwardGeometryPass.hpp"
#include "ShadowMapPass.hpp"
#include "SkyBoxSubPass.hpp"

namespace {
using namespace My;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Error, typename Action>
void Reject(Action action, const char* message) {
    bool rejected = false;
    try {
        action();
    } catch (const Error&) {
        rejected = true;
    }
    Require(rejected, message);
}

class TestApplication final : public IApplication {
   public:
    uint32_t width{16}, height{16};
    uint32_t samples{1};
    void SetCommandLineParameters(int, char**) final {}
    int GetCommandLineArgumentsCount() const final { return 0; }
    const char* GetCommandLineArgument(int) const final { return nullptr; }
    bool IsQuit() const final { return false; }
    void RequestQuit() final {}
    void CreateMainWindow() final { throw std::logic_error("No native window in headless test"); }
    void* GetMainWindowHandler() final { return nullptr; }
    const GfxConfiguration GetConfiguration() const final {
        GfxConfiguration config;
        config.msaaSamples = samples;
        return config;
    }
    void GetFramebufferSize(uint32_t& w, uint32_t& h) final { w = width; h = height; }
};

class TestGraphics final : public VulkanGraphicsManager {
   public:
    using VulkanGraphicsManager::VulkanGraphicsManager;
    using VulkanGraphicsManager::BeginCompute;
    using VulkanGraphicsManager::EndCompute;
    using VulkanGraphicsManager::EndScene;
    using VulkanGraphicsManager::BeginScene;
    using VulkanGraphicsManager::BeginFrame;
    using VulkanGraphicsManager::EndFrame;
    using VulkanGraphicsManager::BeginPass;
    using VulkanGraphicsManager::EndPass;

    Frame& FirstFrame() { return m_Frames.front(); }
    Frame& CurrentFrame() { return m_Frames.at(m_nFrameIndex); }
    const std::vector<Frame>& Frames() const { return m_Frames; }
    void ShareGlobals() {
        for (auto& frame : m_Frames) frame.brdfLUT = m_Frames.front().brdfLUT;
    }
};

void VerifyBrdf(TestGraphics& graphics, const Frame& frame) {
    Require(frame.brdfLUT.handler && frame.brdfLUT.pixel_format == PIXEL_FORMAT::RG32,
            "Shared BRDF pass did not create its engine texture descriptor");
    const auto bytes = graphics.ReadTexture(frame.brdfLUT);
    Require(bytes.size() == 512 * 512 * 4 * sizeof(float),
            "Vulkan BRDF backing does not match the shared rgba32f storage-image ABI");
    for (const auto sample : {std::array<uint32_t, 2>{32, 64}, {255, 255}, {511, 511}}) {
        const auto reference = VulkanTest::ReferenceBrdf(sample[0], sample[1]);
        std::array<float, 4> pixel{};
        std::memcpy(pixel.data(), bytes.data() + (sample[1] * 512 + sample[0]) * sizeof(pixel),
                    sizeof(pixel));
        for (size_t channel = 0; channel < pixel.size(); ++channel) {
            Require(std::isfinite(pixel[channel]) &&
                        std::abs(pixel[channel] - reference[channel % 2]) <= 0.001,
                    "Shared BRDF pass output differs from the independent CPU reference");
        }
    }
}

void CheckBrdf(TestGraphics& graphics, BRDFIntegrator& pass, Frame& frame) {
    pass.BeginPass(frame);
    pass.Dispatch(frame);
    pass.EndPass(frame);
    VerifyBrdf(graphics, frame);
}

class TestScenes final : public ISceneManager {
   public:
    int Initialize() final { return 0; }
    void Finalize() final {}
    void Tick() final {}
    int LoadScene(const char*) final { return 0; }
    uint64_t GetSceneRevision() const final { return 0; }
    const std::shared_ptr<Scene> GetSceneForRendering() const final { return {}; }
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const final { return {}; }
    void ResetScene() final {}
    std::weak_ptr<BaseSceneNode> GetRootNode() const final { return {}; }
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(const std::string&) const final { return {}; }
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(const std::string&) const final { return {}; }
};

void RegisterRendering(VulkanPipelineStateManager& pipelines, uint32_t samples,
                       bool singleFace = false) {
    PipelineState compute;
    compute.pipelineStateName = "PBR BRDF CS";
    compute.pipelineType = PIPELINE_TYPE::COMPUTE;
    compute.computeShaderName = "integrateBRDF.comp";
    compute.flag = PIPELINE_FLAG::NONE;
    Require(pipelines.RegisterPipelineState(compute), "Shared scene BRDF registration failed");
    PipelineState state;
    state.pipelineType = PIPELINE_TYPE::GRAPHIC;
    state.vertexShaderName = "basic.vert";
    state.pixelShaderName = "basic.frag";
    state.pipelineStateName = "BASIC";
    state.depthTestMode = DEPTH_TEST_MODE::LESS_EQUAL;
    state.bDepthWrite = true;
    state.stencilTestMode = STENCIL_TEST_MODE::NONE;
    state.cullFaceMode = CULL_FACE_MODE::BACK;
    state.a2vType = A2V_TYPES::A2V_TYPES_FULL;
    state.flag = PIPELINE_FLAG::NONE;
    state.sampleCount = samples;
    Require(pipelines.RegisterPipelineState(state), "BASIC scene registration failed");
    state.pipelineStateName = "PBR";
    state.vertexShaderName = "pbr.vert";
    state.pixelShaderName = "pbr.frag";
    Require(pipelines.RegisterPipelineState(state), "PBR scene registration failed");
    state.pipelineStateName = "SkyBox";
    state.vertexShaderName = "skybox.vert";
    state.pixelShaderName = "skybox.frag";
    state.a2vType = A2V_TYPES::A2V_TYPES_CUBE;
    state.bDepthWrite = false;
    Require(pipelines.RegisterPipelineState(state), "Skybox scene registration failed");
    state.pipelineStateName = "Omni Light Shadow Map";
    state.vertexShaderName = singleFace ? "shadowmap_omni_single.vert" : "shadowmap_omni.vert";
    state.pixelShaderName = "shadowmap_omni.frag";
    state.geometryShaderName = singleFace ? "" : "shadowmap_omni.geom";
    state.pixelFormat = PIXEL_FORMAT::UNKNOWN;
    state.sampleCount = 1;
    state.a2vType = A2V_TYPES::A2V_TYPES_POS_ONLY;
    state.flag = PIPELINE_FLAG::SHADOW;
    state.cullFaceMode = CULL_FACE_MODE::FRONT;
    state.bDepthWrite = true;
    Require(pipelines.RegisterPipelineState(state), "Omni scene registration failed");
    state.vertexShaderName = "shadowmap.vert";
    state.pixelShaderName = "shadowmap.frag";
    state.geometryShaderName.clear();
    for (const char* name : {"Spot Light Shadow Map", "Area Light Shadow Map", "Sun Light Shadow Map"}) {
        state.pipelineStateName = name;
        Require(pipelines.RegisterPipelineState(state), "Projected shadow registration failed");
    }
}

uint8_t* CopyBytes(const void* data, size_t size) {
    auto* bytes = new uint8_t[size];
    std::memcpy(bytes, data, size);
    return bytes;
}

std::shared_ptr<SceneGeometryNode> AddQuads(
    Scene& scene, const std::string& name, const std::vector<Vector3f>& positions,
    const std::vector<std::string>& materials, bool doubleSided = false) {
    Require(positions.size() == materials.size() * 4, "Invalid quad fixture");
    auto mesh = std::make_shared<SceneObjectMesh>();
    mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeTriList);
    std::vector<float> vertexData;
    for (const auto& position : positions) {
        for (size_t component = 0; component < 3; ++component) vertexData.push_back(position[component]);
    }
    mesh->AddVertexArray(SceneObjectVertexArray("position", 0, VertexDataType::kVertexDataTypeFloat3,
        CopyBytes(vertexData.data(), vertexData.size() * sizeof(float)), vertexData.size()));
    std::vector<float> texcoords;
    for (size_t quad = 0; quad < materials.size(); ++quad) {
        texcoords.insert(texcoords.end(), {0, 0, 1, 0, 1, 1, 0, 1});
    }
    mesh->AddVertexArray(SceneObjectVertexArray("texcoord", 0, VertexDataType::kVertexDataTypeFloat2,
        CopyBytes(texcoords.data(), texcoords.size() * sizeof(float)), texcoords.size()));
    for (uint32_t group = 0; group < materials.size(); ++group) {
        const uint16_t base = static_cast<uint16_t>(group * 4);
        std::vector<uint16_t> indices;
        for (uint16_t index : {0, 1, 2, 0, 2, 3}) indices.push_back(base + index);
        if (doubleSided) {
            for (uint16_t index : {2, 1, 0, 3, 2, 0}) indices.push_back(base + index);
        }
        mesh->AddIndexArray(SceneObjectIndexArray(group, 0, IndexDataType::kIndexDataTypeInt16,
            CopyBytes(indices.data(), indices.size() * sizeof(uint16_t)), indices.size()));
    }
    auto geometry = std::make_shared<SceneObjectGeometry>();
    geometry->SetVisibility(true);
    geometry->SetIfCastShadow(true);
    geometry->AddMesh(std::move(mesh));
    scene.Geometries.emplace(name, geometry);
    auto node = std::make_shared<SceneGeometryNode>(name);
    node->AddSceneObjectRef(name);
    node->SetVisibility(true);
    node->SetIfCastShadow(true);
    for (const auto& material : materials) node->AddMaterialRef(material);
    scene.SceneGraph->AppendChild(node);
    scene.GeometryNodes.emplace(name, node);
    return node;
}

void AddMaterial(Scene& scene, const char* name, Vector4f color) {
    auto material = std::make_shared<SceneObjectMaterial>(name);
    material->SetColor("diffuse", color);
    material->SetParam("metallic", 0);
    material->SetParam("roughness", 0.6f);
    material->SetParam("ao", 1);
    scene.Materials.emplace(name, material);
}

void SetFrame(Frame& frame) {
    frame.clearColor = {0, 0, 0};
    BuildIdentityMatrix(frame.frameContext.viewMatrix);
    BuildIdentityMatrix(frame.frameContext.projectionMatrix);
    frame.frameContext.camPos = {0, 0, 2, 1};
    frame.frameContext.clip_space_type = 1;
    frame.frameContext.numLights = 1;
    frame.lightInfo = {};
    auto& light = frame.lightInfo.lights[0];
    light.lightType = LightType::Omni;
    light.lightPosition = {0, 0, 3, 1};
    light.lightColor = {1, 1, 1, 1};
    light.lightIntensity = 3;
    light.lightShadowMapIndex = -1;
    for (auto& batch : frame.batchContexts) batch->modelMatrix = batch->node->GetRenderTransform();
    // Draw the farther control last so disabling depth testing cannot accidentally pass.
    std::stable_partition(frame.batchContexts.begin(), frame.batchContexts.end(), [](const auto& batch) {
        return batch->node->GetName() != "rear depth control";
    });
}

void Capture(const std::vector<uint8_t>& bytes, uint32_t width, uint32_t height,
             const std::string& name) {
    Require(bytes.size() == size_t{width} * height * 4, "Invalid scene capture dimensions");
    std::filesystem::create_directories("vulkan-images");
    std::ofstream output(std::filesystem::path("vulkan-images") / (name + ".ppm"), std::ios::binary);
    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (size_t offset = 0; offset < bytes.size(); offset += 4) {
        output.write(reinterpret_cast<const char*>(bytes.data() + offset), 3);
    }
    output.flush();
    Require(output.good(), "Cannot write Vulkan offscreen scene artifact");
}

std::vector<uint8_t> RenderScene(TestGraphics& graphics, VulkanPipelineStateManager& pipelines,
                                 bool pbr = true, std::optional<LightType> shadowType = std::nullopt) {
    auto& frame = graphics.CurrentFrame();
    SetFrame(frame);
    if (shadowType) {
        auto& light = frame.lightInfo.lights[0];
        light.lightType = *shadowType;
        light.lightCastShadow = 1;
        light.lightShadowMapIndex = 0;
        light.lightDirection = {0, 0, -1, 0};
        BuildViewRHMatrix(light.lightViewMatrix, {0, 0, 3}, {0, 0, 0}, {0, 1, 0});
        BuildPerspectiveFovRHMatrix(light.lightProjectionMatrix, PI / 2, 1, 1,
                                    POINT_SHADOW_FAR_PLANE);
        if (*shadowType == LightType::Infinity) {
            BuildOrthographicRHMatrix(light.lightProjectionMatrix, -2, 2, 2, -2, 1, 10);
        }
    }
    graphics.BeginFrame(frame);
    if (shadowType) {
        ShadowMapPass shadows(&graphics, &pipelines);
        shadows.BeginPass(frame);
        shadows.Draw(frame);
        shadows.EndPass(frame);
    }
    ForwardGeometryPass forward(&graphics, &pipelines);
    forward.EnableRenderToTexture();
    forward.BeginPass(frame);
    if (pbr) {
        forward.Draw(frame);
    } else {
        graphics.SetPipelineState(pipelines.GetPipelineState("BASIC"), frame);
        graphics.DrawBatch(frame);
    }
    forward.EndPass(frame);
    if (frame.colorTextures.size() > 1) {
        graphics.MSAAResolve(frame.colorTextures[0], frame.colorTextures[1]);
    }
    graphics.EndFrame(frame);
    return graphics.ReadTexture(frame.colorTextures[0]);
}

std::vector<uint8_t> CheckSceneDraws(VulkanRHI& rhi, AssetLoader& assets, uint32_t samples,
                                    const std::vector<uint8_t>& singleSample = {}) {
    TestApplication app;
    app.width = app.height = 64;
    app.samples = samples;
    TestScenes scenes;
    VulkanPipelineStateManager pipelines(rhi, assets);
    RegisterRendering(pipelines, samples);
    TestGraphics graphics(rhi, scenes, pipelines);
    graphics.SetAppPointer(&app);
    Require(graphics.Initialize() == 0, "Scene renderer initialization failed");
    Scene scene("Vulkan scene regression");
    scene.SkyBox.reset();
    AddMaterial(scene, "red", {1, 0.02f, 0.02f, 1});
    AddMaterial(scene, "green", {0.02f, 1, 0.02f, 1});
    AddMaterial(scene, "blue", {0, 0, 1, 1});
    auto gradient = std::make_shared<Image>();
    gradient->Width = gradient->Height = 2;
    gradient->bitcount = 32;
    gradient->bitdepth = 8;
    gradient->pixel_format = PIXEL_FORMAT::RGBA8;
    gradient->pitch = 8;
    gradient->data_size = 16;
    const std::array<uint8_t, 16> texels{255, 5, 0, 255, 255, 5, 0, 255,
                                         128, 5, 0, 255, 128, 5, 0, 255};
    gradient->data = CopyBytes(texels.data(), texels.size());
    scene.Materials.at("red")->SetTexture("diffuse", "generated red gradient");
    scene.Materials.at("red")->GetTextures()[0]->SetTextureImage(gradient);
    auto node = AddQuads(scene, "two materials",
        {{-0.9f, -0.8f, 0.5f}, {-0.1f, -0.8f, 0.5f}, {-0.1f, 0.8f, 0.5f}, {-0.9f, 0.8f, 0.5f},
         {0.1f, -0.8f, 0.5f}, {0.9f, -0.8f, 0.5f}, {0.9f, 0.8f, 0.5f}, {0.1f, 0.8f, 0.5f}},
        {"red", "green"});
    node->SetIfCastShadow(false);
    auto rear = AddQuads(scene, "rear depth control",
        {{-0.9f, -0.8f, 0.75f}, {-0.1f, -0.8f, 0.75f},
         {-0.1f, 0.8f, 0.75f}, {-0.9f, 0.8f, 0.75f}}, {"blue"});
    rear->SetVisibility(false);
    AddQuads(scene, "offscreen shadow blocker",
        {{-0.65f, 0.05f, 1.75f}, {-0.05f, 0.05f, 1.75f},
         {-0.05f, 0.65f, 1.75f}, {-0.65f, 0.65f, 1.75f}}, {"red"}, true);
    graphics.BeginScene(scene);
    std::vector<uint8_t> pbrImage;
    for (bool pbr : {false, true}) {
        const auto bytes = RenderScene(graphics, pipelines, pbr);
        const auto left = (32 * 64 + 16) * 4;
        const auto right = (32 * 64 + 48) * 4;
        std::cout << "Scene samples=" << samples << " PBR=" << pbr << " red="
                  << int(bytes[left]) << ',' << int(bytes[left + 1]) << " green="
                  << int(bytes[right]) << ',' << int(bytes[right + 1]) << '\n';
        Require(bytes[left] > bytes[left + 1] + 30 &&
                    bytes[right + 1] > bytes[right] + 30 &&
                    bytes[left] > 80 && bytes[right + 1] > 80,
                "Production BASIC/PBR failed to render independent material index groups");
        Require(bytes[(32 * 64 + 32) * 4] == 0, "Geometry escaped its transformed bounds/depth target");
        Require(bytes[(16 * 64 + 16) * 4] > bytes[(48 * 64 + 16) * 4] + 15,
                "Material image upload, separate UV stream, or vertical sampling orientation failed");
        Capture(bytes, 64, 64, std::string(pbr ? "pbr-" : "basic-") + std::to_string(samples));
        if (pbr) pbrImage = bytes;
    }
    rear->SetVisibility(true);
    Require(RenderScene(graphics, pipelines) == pbrImage,
            "Farther geometry drawn last overwrote nearer material/depth");
    rear->SetVisibility(false);
    if (samples == 4) {
        Require(rhi.GetPhysicalDevice().getProperties().limits.standardSampleLocations,
                "Analytic MSAA edge fixture requires standard Vulkan sample locations");
        Require(singleSample.size() == pbrImage.size(), "MSAA fixture requires its single-sample control");
        const auto edge = (32 * 64 + 3) * 4;
        Require(std::abs(pbrImage[edge] - singleSample[edge] * 0.75f) <= 3,
                "Four-sample resolve did not produce the analytically expected 3/4 edge coverage");
        const auto interior = (32 * 64 + 16) * 4;
        Require(std::abs(int(pbrImage[interior]) - singleSample[interior]) <= 2,
                "Multisample resolve changed a fully covered interior material");
    }
    if (samples == 1) {
        for (auto type : {LightType::Omni, LightType::Spot, LightType::Area, LightType::Infinity}) {
            const auto shadowed = RenderScene(graphics, pipelines, true, type);
            const auto upperLeft = (16 * 64 + 16) * 4;
            const auto lowerLeft = (48 * 64 + 16) * 4;
            const auto upperRight = (16 * 64 + 48) * 4;
            std::cout << "PBR shadow type=" << static_cast<int>(type) << " shadow="
                      << int(shadowed[upperLeft]) << " lit=" << int(pbrImage[upperLeft]) << '\n';
            Require(shadowed[upperLeft] + 30 < pbrImage[upperLeft] &&
                        std::abs(int(shadowed[lowerLeft]) - pbrImage[lowerLeft]) <= 2 &&
                        std::abs(int(shadowed[upperRight + 1]) - pbrImage[upperRight + 1]) <= 2,
                    "Shared PBR shadow sampling missed the off-axis receiver or inverted its vertical orientation");
            Capture(shadowed, 64, 64, "shadow-" + std::to_string(static_cast<int>(type)));
        }
    }
    node->SetVisibility(false);
    const auto hidden = RenderScene(graphics, pipelines);
    Require(hidden[(32 * 64 + 16) * 4] == 0 && hidden[(32 * 64 + 48) * 4 + 1] == 0,
            "Hidden scene geometry remained visible");
    node->SetVisibility(true);
    node->MoveBy(0, 0.8f, 0);
    const auto moved = RenderScene(graphics, pipelines);
    Require(moved[(48 * 64 + 16) * 4] == 0 && moved[(16 * 64 + 16) * 4] > 80,
            "Per-frame model upload retained the preceding frame transform");
    auto& abandoned = graphics.CurrentFrame();
    SetFrame(abandoned);
    graphics.BeginFrame(abandoned);
    ForwardGeometryPass unfinished(&graphics, &pipelines);
    unfinished.EnableRenderToTexture();
    unfinished.BeginPass(abandoned);
    unfinished.Draw(abandoned);
    graphics.EndScene();
    graphics.BeginScene(scene);
    Require(RenderScene(graphics, pipelines) == moved,
            "Scene reload after abandoned recording changed rendered output");
    graphics.ResizeCanvas(32, 48);
    Require(RenderScene(graphics, pipelines).size() == 32 * 48 * 4,
            "Scene resize did not recreate compatible color/depth/resolve targets");
    graphics.Finalize();
    return pbrImage;
}

void CheckPointShadows(VulkanRHI& rhi, AssetLoader& assets, bool singleFace) {
    TestApplication app;
    TestScenes scenes;
    VulkanPipelineStateManager pipelines(rhi, assets);
    RegisterRendering(pipelines, 1, singleFace);
    TestGraphics graphics(rhi, scenes, pipelines);
    graphics.SetAppPointer(&app);
    Require(graphics.Initialize() == 0, "Shadow renderer initialization failed");
    Scene scene("Vulkan off-axis point shadows");
    scene.SkyBox.reset();
    AddMaterial(scene, "white", {1, 1, 1, 1});
    const std::array<Vector3f, 6> directions{
        Vector3f{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const std::array<Vector3f, 6> right{
        Vector3f{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}};
    const std::array<Vector3f, 6> down{
        Vector3f{0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    std::vector<Vector3f> positions;
    std::vector<std::string> materials;
    for (uint32_t light = 0; light < 2; ++light) {
        for (uint32_t face = 0; face < 6; ++face) {
            const float distance = 2.0f + face * 0.5f + light * 0.25f;
            for (const Vector2f offset : {Vector2f{-0.16f, -0.16f}, {0.16f, -0.16f},
                                          {0.16f, 0.16f}, {-0.16f, 0.16f}}) {
                Vector3f point;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    point[axis] = (axis == 0 ? light * 20.0f : 0.0f) + distance *
                        (directions[face][axis] + right[face][axis] * (0.4f + offset[0]) +
                         down[face][axis] * (-0.2f + offset[1]));
                }
                positions.push_back(point);
            }
            materials.emplace_back("white");
        }
    }
    AddQuads(scene, "twelve unequal off-axis occluders", positions, materials, true);
    graphics.BeginScene(scene);
    auto& frame = graphics.CurrentFrame();
    for (const auto& batch : frame.batchContexts) {
        Require(batch->material.diffuseMap.handler == frame.batchContexts.front()->material.diffuseMap.handler &&
                    batch->material.normalMap.handler == frame.batchContexts.front()->material.normalMap.handler &&
                    batch->material.metallicMap.handler == frame.batchContexts.front()->material.metallicMap.handler,
                "Repeated material references allocated duplicate scalar/image textures per draw group");
    }
    SetFrame(frame);
    frame.frameContext.numLights = 2;
    for (uint32_t light = 0; light < 2; ++light) {
        auto& value = frame.lightInfo.lights[light];
        value.lightType = LightType::Omni;
        value.lightPosition = {light * 20.0f, 0, 0, 1};
        value.lightCastShadow = 1;
        value.lightShadowMapIndex = 1 - light;
        BuildPerspectiveFovRHMatrix(value.lightProjectionMatrix, PI / 2, 1, 1,
                                    POINT_SHADOW_FAR_PLANE);
    }
    graphics.BeginFrame(frame);
    ShadowMapPass pass(&graphics, &pipelines);
    pass.BeginPass(frame);
    pass.Draw(frame);
    pass.EndPass(frame);
    graphics.EndFrame(frame);
    Require(frame.lightInfo.lights[0].lightShadowMapIndex == 1 &&
                frame.lightInfo.lights[1].lightShadowMapIndex == 0,
            "Vulkan shadow recording changed prepared light indices");
    const auto& shadow = frame.frameContext.cubeShadowMap;
    const uint32_t x = static_cast<uint32_t>(shadow.width * 0.7f);
    const uint32_t y = static_cast<uint32_t>(shadow.height * 0.4f);
    const float u = (x + 0.5f) * 2 / shadow.width - 1;
    const float v = (y + 0.5f) * 2 / shadow.height - 1;
    for (uint32_t light = 0; light < 2; ++light) {
        for (uint32_t face = 0; face < 6; ++face) {
            const auto bytes = graphics.ReadTexture(shadow, (1 - light) * 6 + face);
            float depth = 0, center = 0;
            std::memcpy(&depth, bytes.data() + (size_t{y} * shadow.width + x) * sizeof(float), sizeof(float));
            std::memcpy(&center, bytes.data() +
                (size_t{shadow.height / 2} * shadow.width + shadow.width / 2) * sizeof(float), sizeof(float));
            const float expected = (2.0f + face * 0.5f + light * 0.25f) *
                std::sqrt(1 + u * u + v * v) / POINT_SHADOW_FAR_PLANE;
            std::cout << "Point shadow single=" << singleFace << " cube=" << 1 - light
                      << " face=" << face << " depth=" << depth << " expected=" << expected << '\n';
            Require(std::abs(depth - expected) < 0.000002f && center == 1.0f,
                    "Point shadow face orientation/radial depth or distinct cube preservation failed");
        }
    }
    graphics.Finalize();
}

std::shared_ptr<Image> EnvironmentImage(uint32_t index) {
    auto image = std::make_shared<Image>();
    image->Width = image->Height = index >= 6 && index < 12 ? 2 : 4;
    image->bitcount = 32;
    image->bitdepth = 8;
    image->pixel_format = PIXEL_FORMAT::RGBA8;
    image->pitch = image->Width * 4;
    const uint32_t levels = index >= 12 ? 3 : 1;
    for (uint32_t mip = 0; mip < levels; ++mip) {
        const auto width = image->Width >> mip;
        image->mipmaps.emplace_back(width, width, width * 4, image->data_size, width * width * 4);
        image->data_size += width * width * 4;
    }
    image->data = new uint8_t[image->data_size];
    for (uint32_t mip = 0; mip < levels; ++mip) {
        const auto& level = image->mipmaps[mip];
        for (uint32_t y = 0; y < level.Height; ++y) {
            for (uint32_t x = 0; x < level.Width; ++x) {
                const std::array<uint8_t, 4> color{
                    static_cast<uint8_t>(32 + index * 8),
                    static_cast<uint8_t>(64 + (index < 6 ? x : mip) * 16),
                    static_cast<uint8_t>(64 + (index < 6 ? y : mip) * 16), 255};
                std::memcpy(image->data + level.offset + y * level.pitch + x * 4, color.data(), 4);
            }
        }
    }
    return image;
}

void CheckEnvironment(VulkanRHI& rhi, AssetLoader& assets) {
    TestApplication app;
    app.width = app.height = 64;
    TestScenes scenes;
    VulkanPipelineStateManager pipelines(rhi, assets);
    RegisterRendering(pipelines, 1);
    TestGraphics graphics(rhi, scenes, pipelines);
    graphics.SetAppPointer(&app);
    Require(graphics.Initialize() == 0, "Environment renderer initialization failed");
    Scene scene("Vulkan packed environment");
    for (uint32_t index = 0; index < 18; ++index) {
        scene.SkyBox->GetTexture(index).SetTextureImage(EnvironmentImage(index));
    }
    AddMaterial(scene, "white", {1, 1, 1, 1});
    auto node = AddQuads(scene, "environment receiver",
        {{-0.8f, -0.8f, -2}, {0.8f, -0.8f, -2}, {0.8f, 0.8f, -2}, {-0.8f, 0.8f, -2}}, {"white"});
    node->SetVisibility(false);
    graphics.BeginScene(scene);
    const auto sky = graphics.FirstFrame().skybox;
    for (uint32_t face = 0; face < 6; ++face) {
        const auto visible = graphics.ReadTexture(sky, face);
        const auto irradiance = graphics.ReadTexture(sky, face, 1);
        const auto generated = graphics.ReadTexture(sky, face, 2);
        Require(visible[0] == 32 + face * 8 && irradiance[0] == 32 + (face + 6) * 8 &&
                    generated[0] == visible[0],
                "Visible/irradiance packing or generated visible mip content is incorrect");
        for (uint32_t mip = 0; mip < 3; ++mip) {
            const auto radiance = graphics.ReadTexture(sky, face + 6, mip);
            Require(radiance[0] == 32 + (face + 12) * 8 && radiance[1] == 64 + mip * 16,
                    "Packed radiance face/mip upload is incorrect");
        }
    }
    const std::array<Vector3f, 6> directions{
        Vector3f{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const std::array<Vector3f, 6> up{
        Vector3f{0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    ForwardGeometryPass forward(&graphics, &pipelines);
    forward.EnableRenderToTexture();
    for (uint32_t face = 0; face < 6; ++face) {
        auto& frame = graphics.CurrentFrame();
        SetFrame(frame);
        frame.frameContext.numLights = 0;
        BuildViewRHMatrix(frame.frameContext.viewMatrix, {0, 0, 0}, directions[face], up[face]);
        BuildPerspectiveFovRHMatrix(frame.frameContext.projectionMatrix, PI / 2, 1, 1, 10);
        graphics.BeginFrame(frame);
        forward.BeginPass(frame);
        forward.Draw(frame);
        forward.EndPass(frame);
        graphics.EndFrame(frame);
        const auto bytes = graphics.ReadTexture(frame.colorTextures[0]);
        const auto center = (32 * 64 + 32) * 4;
        const float expected = std::pow(1 - std::exp(-(32.0f + face * 8) / 255.0f), 1 / 2.2f) * 255;
        Capture(bytes, 64, 64, "sky-" + std::to_string(face));
        std::cout << "Sky face=" << face << " center=" << int(bytes[center]) << " expected=" << expected
                  << " upperB=" << int(bytes[(16 * 64 + 32) * 4 + 2])
                  << " lowerB=" << int(bytes[(48 * 64 + 32) * 4 + 2]) << '\n';
        Require(std::abs(bytes[center] - expected) <= 1.5f &&
                    bytes[(16 * 64 + 32) * 4 + 2] > bytes[(48 * 64 + 32) * 4 + 2] + 8,
                "Rendered skybox face selection/tone mapping/vertical texture orientation failed");
    }
    node->SetVisibility(true);
    std::array<uint8_t, 2> ambient{};
    for (uint32_t enabled = 0; enabled < 2; ++enabled) {
        auto& frame = graphics.CurrentFrame();
        SetFrame(frame);
        frame.skybox = enabled ? sky : TextureCubeArray{};
        frame.frameContext.numLights = 0;
        frame.frameContext.camPos = {0, 0, 0, 1};
        BuildPerspectiveFovRHMatrix(frame.frameContext.projectionMatrix, PI / 2, 1, 1, 10);
        graphics.BeginFrame(frame);
        forward.BeginPass(frame);
        forward.Draw(frame);
        forward.EndPass(frame);
        graphics.EndFrame(frame);
        const auto bytes = graphics.ReadTexture(frame.colorTextures[0]);
        ambient[enabled] = bytes[(32 * 64 + 32) * 4];
        frame.skybox = sky;
        if (enabled) Capture(bytes, 64, 64, "pbr-environment");
    }
    Require(ambient[0] == 0 && ambient[1] > 40,
            "Shared PBR did not consume packed irradiance/radiance/BRDF textures");
    Texture2D view;
    graphics.CreateTextureView(view, sky, 9, 2);
    const auto selected = graphics.ReadTexture(sky, 9, 2);
    Require(view.width == 1 && view.height == 1 && graphics.ReadTexture(view) == selected,
            "Vulkan GUI cube view did not select its actual face and mip");
    Reject<std::invalid_argument>([&] { graphics.CreateTextureView(view, sky, 0, 0); },
                                  "Vulkan GUI replaced a live texture view");
    Texture2D invalidView;
    Reject<std::invalid_argument>([&] { graphics.CreateTextureView(invalidView, sky, 12, 0); },
                                  "Vulkan GUI accepted an out-of-range cube face");
    graphics.EndScene();
    Require(graphics.ReadTexture(view) == selected,
            "Releasing the scene source invalidated a retained Vulkan GUI view");
    Reject<std::invalid_argument>([&] { graphics.ReadTexture(sky); },
                                  "Vulkan GUI view ownership kept a stale source handle usable");
    graphics.ReleaseTexture(view);
    graphics.Finalize();
}

void CheckSceneInitialization(VulkanRHI& rhi, VulkanPipelineStateManager& pipelines,
                              TestApplication& app) {
    TestScenes scenes;
    TestGraphics graphics(rhi, scenes, pipelines);
    graphics.SetAppPointer(&app);
    Scene scene;
    scene.SkyBox.reset();
    for (int attempt = 0; attempt < 2; ++attempt) {
        Require(graphics.Initialize() == 0, "Scene graphics initialization failed");
        graphics.BeginScene(scene);
        VerifyBrdf(graphics, graphics.FirstFrame());
        const auto first = graphics.FirstFrame();
        for (const auto& frame : graphics.Frames()) {
            Require(frame.brdfLUT.handler == first.brdfLUT.handler &&
                        frame.frameContext.cubeShadowMap.handler ==
                            first.frameContext.cubeShadowMap.handler &&
                        frame.depthTexture.handler == first.depthTexture.handler &&
                        frame.colorTextures.size() == 1 && frame.colorTextures[0].handler,
                    "Shared scene initialization did not preserve framebuffer/global borrowing");
        }
        Require(graphics.Frames()[0].colorTextures[0].handler !=
                    graphics.Frames()[1].colorTextures[0].handler,
                "In-flight frames alias the same offscreen color target");
        graphics.ResizeCanvas(24, 12);
        Reject<std::invalid_argument>([&] { graphics.ReadTexture(first.colorTextures[0]); },
                                      "Resize retained a stale offscreen color allocation");
        Require(graphics.FirstFrame().brdfLUT.handler == first.brdfLUT.handler &&
                    graphics.FirstFrame().colorTextures[0].width == 24 &&
                    graphics.FirstFrame().depthTexture.height == 12,
                "Resize replaced scene globals or failed to update borrowed targets");
        graphics.EndScene();
        Reject<std::invalid_argument>([&] { graphics.ReadTexture(first.frameContext.cubeShadowMap); },
                                      "Scene teardown retained a shared shadow-array allocation");
        graphics.Finalize();
        graphics.Finalize();
    }
}

void CheckTextures(TestGraphics& graphics) {
    Texture2D color;
    color.width = color.height = 16;
    color.pixel_format = PIXEL_FORMAT::RGBA8;
    graphics.GenerateTexture(color);
    const auto borrowed = color;
    Reject<std::invalid_argument>([&] { graphics.GenerateTexture(color); },
                                  "Texture generation replaced a live owned texture");
    graphics.ReleaseTexture(color);
    graphics.ReleaseTexture(color);
    Reject<std::invalid_argument>([&] { graphics.ReadTexture(borrowed); },
                                  "A stale borrowed texture descriptor remained usable");

    Texture2D invalid;
    invalid.width = 0;
    invalid.height = 16;
    invalid.pixel_format = PIXEL_FORMAT::RGBA8;
    Reject<std::invalid_argument>([&] { graphics.GenerateTexture(invalid); },
                                  "Invalid image dimensions were accepted");
    Require(!invalid.handler, "Failed texture creation published a handle");
    invalid.width = 16;
    invalid.samples = 3;
    Reject<std::invalid_argument>([&] { graphics.GenerateTexture(invalid); },
                                  "Invalid multisample allocation was silently downgraded");

    Texture2D depth;
    depth.width = depth.height = 16;
    depth.pixel_format = PIXEL_FORMAT::D32;
    graphics.GenerateTexture(depth);
    graphics.ReleaseTexture(depth);

    Texture2DArray array;
    array.width = array.height = 16;
    array.size = 2;
    array.pixel_format = PIXEL_FORMAT::D32;
    graphics.GenerateTextureArray(array);
    graphics.ReleaseTexture(array);
    TextureCubeArray cubes;
    cubes.width = cubes.height = 16;
    cubes.size = 2;
    cubes.pixel_format = PIXEL_FORMAT::D32;
    graphics.GenerateCubemapArray(cubes);
    graphics.ReleaseTexture(cubes);
}
}  // namespace

int main(int argc, char** argv) {
    using namespace My;
    try {
        Require(argc == 2, "Expected generated asset root");
        VulkanRHI rhi;
        rhi.InitializeHeadless();
        {
            TestApplication app;
            AssetLoader assets;
            assets.AddSearchPath(argv[1]);
            VulkanPipelineStateManager pipelines(rhi, assets);
            PipelineState compute;
            compute.pipelineStateName = "PBR BRDF CS";
            compute.pipelineType = PIPELINE_TYPE::COMPUTE;
            compute.computeShaderName = "integrateBRDF.comp";
            compute.flag = PIPELINE_FLAG::NONE;
            Require(pipelines.RegisterPipelineState(compute), "BRDF registration failed");
            TestGraphics graphics(rhi);
            graphics.SetAppPointer(&app);
            Require(graphics.Initialize() == 0, "Graphics initialization failed");
            BRDFIntegrator pass(&graphics, &pipelines);
            auto& frame = graphics.FirstFrame();
            Reject<std::logic_error>([&] { graphics.Dispatch(512, 512, 1); },
                                     "Dispatch outside a shared compute pass was accepted");
            graphics.BeginCompute();
            Reject<std::logic_error>([&] { graphics.BeginCompute(); },
                                     "Nested compute pass was accepted");
            graphics.SetPipelineState(pipelines.GetPipelineState("PBR BRDF CS"), frame);
            Reject<std::logic_error>([&] { graphics.Dispatch(512, 512, 1); },
                                     "Dispatch without storage output was accepted");
            Texture2D small;
            small.width = small.height = 16;
            small.pixel_format = PIXEL_FORMAT::RG32;
            graphics.GenerateTextureForWrite(small);
            Reject<std::invalid_argument>([&] { graphics.BindTextureForWrite(small, 1); },
                                          "Unsupported storage slot was accepted");
            graphics.BindTextureForWrite(small, 0);
            Reject<std::invalid_argument>([&] { graphics.Dispatch(512, 512, 1); },
                                          "BRDF dispatch could overrun its storage output");
            graphics.EndCompute();
            graphics.ReleaseTexture(small);

            CheckBrdf(graphics, pass, frame);
            const auto firstHandle = frame.brdfLUT.handler;
            CheckBrdf(graphics, pass, frame);
            Require(frame.brdfLUT.handler == firstHandle,
                    "Repeated shared BRDF dispatch replaced its texture");
            CheckTextures(graphics);
            graphics.ShareGlobals();
            const auto staleBrdf = frame.brdfLUT;
            Texture2D independent;
            independent.width = independent.height = 16;
            independent.pixel_format = PIXEL_FORMAT::RGBA8;
            graphics.GenerateTexture(independent);
            graphics.EndScene();
            graphics.ReleaseTexture(independent);
            Reject<std::invalid_argument>([&] { graphics.ReadTexture(staleBrdf); },
                                          "Scene teardown retained a stale BRDF allocation");
            frame = Frame{};
            CheckBrdf(graphics, pass, frame);
            Require(frame.brdfLUT.handler != firstHandle,
                    "Scene reload reused a stale texture identity");
            graphics.Finalize();
            graphics.Finalize();
            Reject<std::logic_error>([&] { graphics.BeginCompute(); },
                                     "Finalized graphics manager accepted a compute pass");
            Require(graphics.Initialize() == 0, "Graphics reinitialization failed");
            frame = Frame{};
            CheckBrdf(graphics, pass, frame);
            CheckSceneInitialization(rhi, pipelines, app);
            const auto singleSampleScene = CheckSceneDraws(rhi, assets, 1);
            CheckSceneDraws(rhi, assets, 4, singleSampleScene);
            CheckPointShadows(rhi, assets, false);
            CheckPointShadows(rhi, assets, true);
            CheckEnvironment(rhi, assets);
            graphics.BeginCompute();
            graphics.SetPipelineState(pipelines.GetPipelineState("PBR BRDF CS"), frame);
            graphics.BindTextureForWrite(frame.brdfLUT, 0);
            pipelines.Finalize();
            Reject<std::logic_error>([&] { graphics.Dispatch(512, 512, 1); },
                                     "A finalized pipeline registry remained dispatchable");
            graphics.EndCompute();
            graphics.Finalize();
        }
        rhi.destroyAll();
        Require(rhi.GetValidationErrorCount() == 0, "Graphics integration produced validation errors");
        std::cout << "Shared GraphicsManager BRDF, BASIC/PBR materials and UVs, per-frame uniforms, "
                     "six-face multi-light shadows and receivers, skybox/IBL, MSAA resolve, "
                     "abandoned recording, resize, and resource lifecycle passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan graphics integration failed: " << error.what() << '\n';
        return 1;
    }
}

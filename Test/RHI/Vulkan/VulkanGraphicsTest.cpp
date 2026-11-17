#include "Vulkan/VulkanGraphicsManager.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "AssetLoader.hpp"
#include "BRDFIntegrator.hpp"
#include "BrdfReference.hpp"
#include "ISceneManager.hpp"

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
    void SetCommandLineParameters(int, char**) final {}
    int GetCommandLineArgumentsCount() const final { return 0; }
    const char* GetCommandLineArgument(int) const final { return nullptr; }
    bool IsQuit() const final { return false; }
    void RequestQuit() final {}
    void CreateMainWindow() final { throw std::logic_error("No native window in headless test"); }
    void* GetMainWindowHandler() final { return nullptr; }
    const GfxConfiguration GetConfiguration() const final {
        GfxConfiguration config;
        config.msaaSamples = 1;
        return config;
    }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) final { width = height = 16; }
};

class TestGraphics final : public VulkanGraphicsManager {
   public:
    using VulkanGraphicsManager::VulkanGraphicsManager;
    using VulkanGraphicsManager::BeginCompute;
    using VulkanGraphicsManager::EndCompute;
    using VulkanGraphicsManager::EndScene;
    using VulkanGraphicsManager::BeginScene;

    Frame& FirstFrame() { return m_Frames.front(); }
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
    invalid.samples = 4;
    Reject<std::invalid_argument>([&] { graphics.GenerateTexture(invalid); },
                                  "Unimplemented multisample allocation was silently downgraded");

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
        std::cout << "Shared GraphicsManager BRDF pass and owned texture lifecycle passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan graphics integration failed: " << error.what() << '\n';
        return 1;
    }
}

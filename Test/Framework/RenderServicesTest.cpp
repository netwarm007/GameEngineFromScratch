#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "GraphicsManager.hpp"
#include "ISceneManager.hpp"
#include "imgui.h"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class TestApplication : public IApplication {
   public:
    void SetCommandLineParameters(int, char**) override {}
    int GetCommandLineArgumentsCount() const override { return 0; }
    const char* GetCommandLineArgument(int) const override { return nullptr; }
    bool IsQuit() const override { return false; }
    void RequestQuit() override {}
    void CreateMainWindow() override {}
    void* GetMainWindowHandler() override { return nullptr; }
    const GfxConfiguration GetConfiguration() const override { return config; }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) override {
        width = 640;
        height = 480;
    }

    GfxConfiguration config;
};

class TestScenes : public ISceneManager {
   public:
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    int LoadScene(const char*) override { return 0; }
    uint64_t GetSceneRevision() const override { return revision; }
    const std::shared_ptr<Scene> GetSceneForRendering() const override {
        ++reads;
        return scene;
    }
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const override {
        return scene;
    }
    void ResetScene() override {}
    std::weak_ptr<BaseSceneNode> GetRootNode() const override { return {}; }
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(
        const std::string&) const override {
        return {};
    }
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(
        const std::string&) const override {
        return {};
    }

    uint64_t revision = 0;
    std::shared_ptr<Scene> scene;
    mutable int reads = 0;
};

class TestPipelines : public IPipelineStateManager {
   public:
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    bool RegisterPipelineState(PipelineState&) override { return true; }
    void UnregisterPipelineState(PipelineState&) override {}
    void Clear() override {}
    const std::shared_ptr<PipelineState> GetPipelineState(
        std::string name) const override {
        requests.push_back(name);
        return pipeline;
    }

    std::shared_ptr<PipelineState> pipeline = std::make_shared<PipelineState>();
    mutable std::vector<std::string> requests;
};

class RecordingGraphics : public GraphicsManager {
   public:
    using GraphicsManager::GraphicsManager;

    void Draw() override {
        events.push_back("draw");
        drawnScene = m_Frames[m_nFrameIndex].scene.get();
    }
    void Present() override { events.push_back("present"); }
    void SetPipelineState(const std::shared_ptr<PipelineState>& pipeline,
                          const Frame&) override {
        boundPipeline = pipeline;
    }
    void Dispatch(uint32_t, uint32_t, uint32_t) override { ++dispatches; }

    bool HasScenePasses() const {
        return !m_InitPasses.empty() || !m_DrawPasses.empty();
    }
    bool HasFrameScene() const {
        for (const auto& frame : m_Frames) {
            if (frame.scene) return true;
        }
        return false;
    }

    const Scene* begunScene = nullptr;
    const Scene* drawnScene = nullptr;
    std::shared_ptr<PipelineState> boundPipeline;
    std::vector<std::string> events;
    int dispatches = 0;
    int sceneEnds = 0;

   protected:
    void BeginScene(const Scene& scene) override {
        events.push_back("begin scene");
        begunScene = &scene;
        GraphicsManager::BeginScene(scene);
    }
    void EndScene() override {
        events.push_back("end scene");
        ++sceneEnds;
        GraphicsManager::EndScene();
    }
    void BeginFrame(Frame&) override { events.push_back("begin frame"); }
    void EndFrame(Frame&) override { events.push_back("end frame"); }
};

class FrameResourceGraphics : public RecordingGraphics {
   public:
    using RecordingGraphics::RecordingGraphics;

    void GenerateTexture(Texture2D& texture) override {
        texture.handler = ++allocations;
        if (bufferHandles) {
            texture.buff = texture.handler;
            texture.handler = 0;
        }
        liveTextures.insert(allocations);
    }
    void ReleaseTexture(TextureBase& texture) override {
        const auto id = texture.buff ? texture.buff : texture.handler;
        Expect(liveTextures.erase(id) == 1,
               "only the owner releases a live framebuffer texture, exactly once");
        ++releases;
        // OpenGL and Metal do not clear the caller's texture descriptor.
    }
    const std::vector<Frame>& Frames() const { return m_Frames; }

    bool bufferHandles = false;
    TextureHandler allocations = 0;
    TextureHandler releases = 0;
    std::set<TextureHandler> liveTextures;
};

void ExpectFrameTargets(const FrameResourceGraphics& graphics, uint32_t samples,
                        uint32_t width, uint32_t height) {
    std::set<TextureHandler> colors;
    const auto& frames = graphics.Frames();
    const auto depth = frames[0].depthTexture;
    for (const auto& frame : frames) {
        Expect(frame.colorTextures.size() == (samples > 1 ? 2u : 1u),
               "each frame borrows one color target and an optional MSAA target");
        if (frame.colorTextures.empty()) continue;
        const auto& color = frame.colorTextures[0];
        colors.insert(color.buff ? color.buff : color.handler);
        Expect(color.width == width && color.height == height &&
                   color.samples == 1 && color.pixel_format == PIXEL_FORMAT::RGBA8,
               "per-frame color targets preserve dimensions and format");
        Expect(frame.depthTexture.handler == depth.handler &&
                   frame.depthTexture.buff == depth.buff &&
                   depth.width == width && depth.height == height &&
                   depth.samples == samples &&
                   depth.pixel_format == PIXEL_FORMAT::D32,
               "every frame borrows the same correctly sized depth target");
        Expect(frame.enableMSAA == (samples > 1),
               "frame MSAA state reflects the current configuration");
        if (samples > 1 && frame.colorTextures.size() == 2) {
            const auto& msaa = frame.colorTextures[1];
            Expect(msaa.handler == frames[0].colorTextures[1].handler &&
                       msaa.buff == frames[0].colorTextures[1].buff &&
                       msaa.width == width && msaa.height == height &&
                       msaa.samples == samples &&
                       msaa.pixel_format == PIXEL_FORMAT::RGBA8,
                   "every frame borrows the same correctly sized MSAA target");
        }
    }
    Expect(colors.size() == frames.size(),
           "each in-flight frame has its own single-sample color target");
}

void TestFrameResourceOwnership(uint32_t samples, bool bufferHandles) {
    TestApplication app;
    app.config.msaaSamples = samples;
    TestScenes scenes;
    TestPipelines pipelines;
    FrameResourceGraphics graphics(scenes, pipelines);
    graphics.bufferHandles = bufferHandles;
    graphics.SetAppPointer(&app);
    Expect(graphics.Initialize() == 0, "frame resource coordinator initializes");
    Expect(graphics.liveTextures.empty(),
           "initialization does not allocate offscreen targets before a scene");
    scenes.revision = 1;
    scenes.scene = std::make_shared<Scene>("frame resources");
    graphics.Tick();

    const auto targetCount =
        GfxConfiguration::kMaxInFlightFrameCount + (samples > 1 ? 2u : 1u);
    Expect(graphics.liveTextures.size() == targetCount && graphics.releases == 0,
           "one owned target exists per color plus shared depth and optional MSAA");
    ExpectFrameTargets(graphics, samples, 640, 480);

    graphics.ResizeCanvas(800, 600);
    Expect(graphics.releases == targetCount &&
               graphics.liveTextures.size() == targetCount,
           "resize releases each old target once and replaces the full target set");
    ExpectFrameTargets(graphics, samples, 800, 600);
    const auto allocations = graphics.allocations;
    graphics.ResizeCanvas(800, 600);
    graphics.ResizeCanvas(0, 0);
    Expect(graphics.allocations == allocations,
           "unchanged and minimized canvases keep their targets");

    app.config.msaaSamples = samples > 1 ? 1 : 4;
    graphics.ResizeCanvas(1024, 768);
    ExpectFrameTargets(graphics, app.config.msaaSamples, 1024, 768);

    std::weak_ptr<Scene> oldScene = scenes.scene;
    scenes.scene = std::make_shared<Scene>("replacement frame resources");
    ++scenes.revision;
    graphics.Tick();
    const auto replacementCount = GfxConfiguration::kMaxInFlightFrameCount +
                                 (app.config.msaaSamples > 1 ? 2u : 1u);
    Expect(oldScene.expired() &&
               graphics.liveTextures.size() == replacementCount &&
               graphics.allocations - graphics.releases == replacementCount,
           "scene replacement retires old targets and snapshots without leaks");
    ExpectFrameTargets(graphics, app.config.msaaSamples, 1024, 768);

    graphics.Finalize();
    Expect(graphics.liveTextures.empty() &&
               graphics.releases == graphics.allocations,
           "finalization releases the final targets exactly once");
    for (const auto& frame : graphics.Frames()) {
        Expect(frame.colorTextures.empty() && !frame.depthTexture.handler &&
                   !frame.depthTexture.buff && !frame.enableMSAA && !frame.scene,
               "finalization invalidates borrowed targets and scene snapshots");
    }
    const auto finalAllocations = graphics.allocations;
    const auto finalReleases = graphics.releases;
    graphics.Finalize();
    graphics.ResizeCanvas(1280, 720);
    Expect(graphics.allocations == finalAllocations &&
               graphics.releases == finalReleases,
           "repeated cleanup and post-finalization resize do not touch GPU resources");
}

void TestFrameResourcesAfterFailedInitialization() {
    FrameResourceGraphics graphics;
    Expect(graphics.Initialize() != 0,
           "frame initialization without a platform application fails");
    graphics.Finalize();
    graphics.Finalize();
    graphics.ResizeCanvas(640, 480);
    Expect(graphics.allocations == 0 && graphics.releases == 0,
           "failed initialization is safe to clean up and allocates no targets");
}

void TestInjectedServices() {
    TestApplication app;
    TestScenes scenes;
    TestPipelines pipelines;
    RecordingGraphics graphics(scenes, pipelines);
    graphics.SetAppPointer(&app);
    Expect(graphics.Initialize() == 0,
           "render services initialize with IApplication, not BaseApplication");
    Expect(graphics.HasScenePasses(), "injected pipeline service creates passes");

    graphics.Tick();
    Expect(graphics.events.empty() && scenes.reads == 0,
           "unloaded scenes do not begin rendering or fetch a snapshot");

    scenes.revision = 1;
    scenes.scene = std::make_shared<Scene>("first");
    graphics.Tick();
    Expect(graphics.events == std::vector<std::string>{
               "end scene", "begin scene", "begin frame", "draw",
               "end frame", "present"},
           "scene preparation and rendering preserve their order");
    Expect(scenes.reads == 1 && graphics.begunScene == scenes.scene.get() &&
               graphics.drawnScene == scenes.scene.get(),
           "scene setup and frame rendering share one injected snapshot");
#if !defined(OS_WEBASSEMBLY)
    Expect(pipelines.requests == std::vector<std::string>{"PBR BRDF CS"} &&
               graphics.boundPipeline == pipelines.pipeline &&
               graphics.dispatches == 1,
           "initial pass uses the injected pipeline and backend instance");
#endif

    graphics.events.clear();
    graphics.Tick();
    Expect(scenes.reads == 2 && graphics.events == std::vector<std::string>{
               "begin frame", "draw", "end frame", "present"},
           "unchanged revision updates the frame without rebuilding resources");

    const int sceneEnds = graphics.sceneEnds;
    std::weak_ptr<Scene> oldScene = scenes.scene;
    scenes.revision = 2;
    scenes.scene.reset();
    graphics.events.clear();
    graphics.Tick();
    Expect(graphics.events.empty() && graphics.sceneEnds == sceneEnds &&
               !oldScene.expired(),
           "missing snapshot skips rendering without discarding the old scene");

    scenes.scene = std::make_shared<Scene>("replacement");
    graphics.Tick();
    Expect(graphics.sceneEnds == sceneEnds + 1 && oldScene.expired() &&
               graphics.begunScene == scenes.scene.get() &&
               graphics.drawnScene == scenes.scene.get(),
           "valid replacement retries the revision and releases the old snapshot");
    graphics.Finalize();
    Expect(!graphics.HasFrameScene(), "finalization releases every frame snapshot");
}

void TestGraphicsOnlyMode() {
    TestApplication app;
    RecordingGraphics graphics;
    Expect(graphics.Initialize() != 0,
           "missing platform application fails initialization explicitly");
    graphics.SetAppPointer(&app);
    Expect(graphics.Initialize() == 0 && !graphics.HasScenePasses(),
           "graphics-only clients need no scene or pipeline service");
    graphics.Tick();
    Expect(graphics.events == std::vector<std::string>{
               "begin frame", "draw", "end frame", "present"} &&
               !graphics.HasFrameScene(),
           "graphics-only draw loops still render without a loaded scene");
    graphics.Finalize();
}
}  // namespace

int main() {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(640, 480);
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    TestInjectedServices();
    TestGraphicsOnlyMode();
    TestFrameResourceOwnership(1, false);
    TestFrameResourceOwnership(4, false);
    TestFrameResourceOwnership(4, true);
    TestFrameResourcesAfterFailedInitialization();

    ImGui::DestroyContext();
    return failures == 0 ? 0 : 1;
}

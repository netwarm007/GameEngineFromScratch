#include <iostream>
#include <memory>
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

    ImGui::DestroyContext();
    return failures == 0 ? 0 : 1;
}

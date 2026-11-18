#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ISceneManager.hpp"
#include "Viewer/ViewerLogic.hpp"

using namespace My;

namespace {
int failures = 0;
void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool SetMarkerPath(const std::filesystem::path& path) {
#if defined(_WIN32)
    return _putenv_s("GEFS_SCREENSHOT_WAIT_FOR", path.string().c_str()) == 0;
#else
    return setenv("GEFS_SCREENSHOT_WAIT_FOR", path.string().c_str(), 1) == 0;
#endif
}

class TestApplication : public IApplication {
   public:
    std::vector<std::string> args{"Viewer"};
    bool quit = false;
    void SetCommandLineParameters(int, char**) override {}
    int GetCommandLineArgumentsCount() const override {
        return static_cast<int>(args.size());
    }
    const char* GetCommandLineArgument(int i) const override {
        return args.at(i).c_str();
    }
    bool IsQuit() const override { return quit; }
    void RequestQuit() override { quit = true; }
    void CreateMainWindow() override {}
    void* GetMainWindowHandler() override { return nullptr; }
    const GfxConfiguration GetConfiguration() const override {
        return GfxConfiguration(8, 8, 8, 8, 24, 8, 1, 800, 600, "test");
    }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) override {
        width = 800;
        height = 600;
    }
};

class TestScenes : public ISceneManager {
   public:
    std::vector<std::string> loads;
    int result = 0;
    std::shared_ptr<Scene> scene;
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    int LoadScene(const char* path) override {
        loads.emplace_back(path);
        return result;
    }
    uint64_t GetSceneRevision() const override { return 0; }
    const std::shared_ptr<Scene> GetSceneForRendering() const override { return scene; }
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const override { return scene; }
    void ResetScene() override {}
    std::weak_ptr<BaseSceneNode> GetRootNode() const override { return {}; }
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(const std::string&) const override { return {}; }
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(const std::string&) const override { return {}; }
};

void TestLoadingAndCycle() {
    TestApplication app;
    TestScenes scenes;
    ViewerLogic viewer(scenes);
    IGameLogic& logic = viewer;
    Expect(logic.Initialize() != 0, "missing application is rejected explicitly");
    logic.SetAppPointer(&app);
    Expect(logic.Initialize() == 0 &&
               scenes.loads == std::vector<std::string>{"Scene/splash.ogex"},
           "plain IApplication loads splash through injected scene service");
    app.args = {"Viewer", "-screenshot", "capture.png", "Scene/custom.ogex"};
    scenes.result = -7;
    Expect(logic.Initialize() == -7 && scenes.loads.back() == "Scene/custom.ogex",
           "custom scene skips screenshot options and propagates load failure");
    scenes.result = 0;
    for (const auto& args : {std::vector<std::string>{"Viewer", "-sceneCycle"},
                            std::vector<std::string>{"Viewer", "-sceneCycle", "3"}}) {
        app.args = args;
        const auto count = scenes.loads.size();
        Expect(logic.Initialize() != 0 && scenes.loads.size() == count,
               "invalid scene-cycle argument does not load a scene");
    }
    for (const auto* phase : {"1", "2"}) {
        app.args = {"Viewer", "-sceneCycle", phase};
        scenes.loads.clear();
        Expect(logic.Initialize() == 0, "cycle initializes");
        for (int tick = 0; tick < 24; ++tick) logic.Tick();
        std::vector<std::string> expected{"Scene/splash.ogex", "Scene/billiard.ogex"};
        if (std::string(phase) == "2") expected.push_back("Scene/splash.ogex");
        Expect(scenes.loads == expected, "cycle preserves A/B and A/B/A loading order");
        logic.Finalize();
    }
    app.args = {"Viewer", "-sceneCycle", "2"};
    logic.Initialize();
    scenes.result = -1;
    for (int tick = 0; tick < 8; ++tick) logic.Tick();
    Expect(app.quit, "cycle load failure requests quit through IApplication");
    scenes.result = 0;
    app.quit = false;
    app.args = {"Viewer"};
    scenes.loads.clear();
    logic.Initialize();
    for (int tick = 0; tick < 24; ++tick) logic.Tick();
    Expect(scenes.loads.size() == 1 && !app.quit,
           "new ordinary lifecycle does not retain previous cycle state");
}

void TestSceneCycleMarker() {
    const auto tempDir = std::filesystem::temp_directory_path() /
                         ("gefs-viewer-marker-" +
                          std::to_string(std::chrono::steady_clock::now()
                                             .time_since_epoch()
                                             .count()));
    std::filesystem::create_directories(tempDir);
    const auto marker = tempDir / "ready.txt";

    TestApplication app;
    TestScenes scenes;
    ViewerLogic viewer(scenes);
    IGameLogic& logic = viewer;
    logic.SetAppPointer(&app);
    app.args = {"Viewer", "-sceneCycle", "1"};

    {
        std::ofstream staleMarker(marker);
        staleMarker << "stale";
    }
    Expect(std::filesystem::exists(marker), "stale marker fixture is created");
    Expect(SetMarkerPath(marker), "scene-cycle marker path is configurable");
    Expect(logic.Initialize() == 0 && !std::filesystem::exists(marker),
           "initialization clears a stale scene-cycle marker");
    for (int tick = 0; tick < 15; ++tick) logic.Tick();
    Expect(!std::filesystem::exists(marker),
           "scene-cycle marker is not written before the ready tick");
    logic.Tick();
    std::ifstream readyMarker(marker);
    std::string phase;
    std::getline(readyMarker, phase);
    Expect(readyMarker.is_open(),
           "scene-cycle marker is readable after the ready tick");
    Expect(phase == "1", "scene-cycle marker contains the completed phase");
    readyMarker.close();
    std::filesystem::remove(marker);

    const auto nonEmptyDirectory = tempDir / "not-a-marker";
    std::filesystem::create_directories(nonEmptyDirectory);
    {
        std::ofstream sentinel(nonEmptyDirectory / "sentinel");
        sentinel << "keep";
    }
    const auto loadsBeforeClearFailure = scenes.loads.size();
    Expect(SetMarkerPath(nonEmptyDirectory), "marker error path is configurable");
    Expect(logic.Initialize() != 0 &&
               scenes.loads.size() == loadsBeforeClearFailure,
           "marker cleanup failure rejects initialization before loading a scene");

    const auto blockedMarker = tempDir / "blocked-marker";
    Expect(SetMarkerPath(blockedMarker), "marker write failure path is configurable");
    Expect(logic.Initialize() == 0, "marker write failure case initializes");
    std::filesystem::create_directory(blockedMarker);
    for (int tick = 0; tick < 16; ++tick) logic.Tick();
    Expect(app.quit, "marker write failure requests application quit");
    std::filesystem::remove_all(tempDir);
}

std::shared_ptr<SceneCameraNode> SetCamera(TestScenes& scenes) {
    scenes.scene = std::make_shared<Scene>("camera");
    auto node = std::make_shared<SceneCameraNode>("camera node");
    node->MoveBy(0, -5, 0);
    node->SetTarget({0, 0, 0});
    scenes.scene->SceneGraph->AppendChild(node);
    scenes.scene->CameraNodes.emplace("camera", node);
    return node;
}

void TestCameraInput() {
    TestScenes scenes;
    ViewerLogic viewer(scenes);
    IGameLogic& logic = viewer;
    auto node = SetCamera(scenes);
    logic.OnRightKeyDown();
    logic.OnUpKeyDown();
    auto target = node->GetTarget();
    Expect(target[0] == 1 && target[1] == 1 && target[2] == 0,
           "right/up move camera target along local positive axes");
    auto transform = node->GetCalculatedTransform();
    Expect((*transform)[3][0] == 1 && (*transform)[3][1] == -4,
           "camera position moves with target");
    logic.OnLeftKeyDown();
    logic.OnDownKeyDown();
    target = node->GetTarget();
    Expect(target[0] == 0 && target[1] == 0, "left/down reverse movement");
    auto replacement = SetCamera(scenes);
    logic.OnRightKeyDown();
    Expect(node->GetTarget()[0] == 0 && replacement->GetTarget()[0] == 1,
           "input resolves current injected scene rather than retaining old camera");
    auto expected = std::make_shared<SceneCameraNode>("expected");
    expected->MoveBy(0, -5, 0);
    expected->MoveBy(1, 0, 0);
    logic.OnAnalogStick(0, 90, -45);
    logic.OnAnalogStick(1, 90, -45);
    expected->RotateBy(-45 / (-1800 * 2.0 * PI), 0, 0);
    expected->RotateBy(0, 0, 90 / (-1800 * 2.0 * PI));
    auto actualMatrix = replacement->GetCalculatedTransform();
    auto expectedMatrix = expected->GetCalculatedTransform();
    bool equal = true;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            equal &= std::abs((*actualMatrix)[row][col] -
                              (*expectedMatrix)[row][col]) < 0.00001f;
    Expect(equal, "right drag preserves rotation signs/order and left drag is ignored");
    scenes.scene = std::make_shared<Scene>("no camera");
    logic.OnLeftKeyDown();
    logic.OnRightKeyDown();
    scenes.scene.reset();
    logic.OnUpKeyDown();
    logic.OnDownKeyDown();
    logic.OnAnalogStick(1, 1, 1);
}
}  // namespace

int main() {
    const char* existing = std::getenv("GEFS_SCREENSHOT_WAIT_FOR");
    const bool hadMarker = existing != nullptr;
    const std::string savedMarker = existing ? existing : "";
#if defined(_WIN32)
    if (_putenv_s("GEFS_SCREENSHOT_WAIT_FOR", "") != 0) return 1;
#else
    if (unsetenv("GEFS_SCREENSHOT_WAIT_FOR") != 0) return 1;
#endif
    TestLoadingAndCycle();
    TestSceneCycleMarker();
    TestCameraInput();
#if defined(_WIN32)
    if (_putenv_s("GEFS_SCREENSHOT_WAIT_FOR", savedMarker.c_str()) != 0) return 1;
#else
    if (hadMarker ? setenv("GEFS_SCREENSHOT_WAIT_FOR", savedMarker.c_str(), 1) != 0
                  : unsetenv("GEFS_SCREENSHOT_WAIT_FOR") != 0) return 1;
#endif
    return failures ? 1 : 0;
}

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "Game/Billiard/BilliardGameLogic.hpp"
#include "IPhysicsManager.hpp"
#include "ISceneManager.hpp"

using namespace My;

namespace {
int failures = 0;
void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class TestScenes : public ISceneManager {
   public:
    int loadResult = 0;
    std::vector<std::string> loads;
    mutable std::vector<std::string> geometryLookups;
    std::weak_ptr<SceneGeometryNode> cue;
    std::shared_ptr<Scene> scene;

    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    int LoadScene(const char* path) override {
        loads.emplace_back(path);
        return loadResult;
    }
    uint64_t GetSceneRevision() const override { return 0; }
    const std::shared_ptr<Scene> GetSceneForRendering() const override {
        return scene;
    }
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const override {
        return scene;
    }
    void ResetScene() override {}
    std::weak_ptr<BaseSceneNode> GetRootNode() const override { return {}; }
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(
        const std::string& name) const override {
        geometryLookups.push_back(name);
        return name == "pbb_cue" ? cue : std::weak_ptr<SceneGeometryNode>{};
    }
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(
        const std::string&) const override { return {}; }
};

class TestPhysics : public IPhysicsManager {
   public:
    struct Force {
        SceneGeometryNode* node;
        Vector3f value;
    };
    std::vector<Force> forces;
    bool hasBody = true;

    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    void CreateRigidBody(const std::shared_ptr<SceneGeometryNode>&,
                         const SceneObjectGeometry&) override {}
    void DeleteRigidBody(SceneGeometryNode&) override {}
    int CreateRigidBodies() override { return 0; }
    void ClearRigidBodies() override {}
    bool UpdateRigidBodyTransform(SceneGeometryNode&) override { return false; }
    bool ApplyCentralForce(SceneGeometryNode& node, Vector3f force) override {
        forces.push_back({&node, force});
        return hasBody;
    }
};

class TestApplication : public IApplication {
   public:
    GfxConfiguration config{8, 8, 8, 8, 24, 8, 1, 800, 600, "test"};

    void SetCommandLineParameters(int, char**) override {}
    int GetCommandLineArgumentsCount() const override { return 0; }
    const char* GetCommandLineArgument(int) const override { return nullptr; }
    bool IsQuit() const override { return false; }
    void RequestQuit() override {}
    void CreateMainWindow() override {}
    void* GetMainWindowHandler() override { return nullptr; }
    const GfxConfiguration GetConfiguration() const override { return config; }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) override {
        width = config.screenWidth;
        height = config.screenHeight;
    }
};

bool SameTransform(const SceneCameraNode& actual, const SceneCameraNode& expected) {
    const auto a = actual.GetCalculatedTransform();
    const auto e = expected.GetCalculatedTransform();
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            if (!(std::abs((*a)[row][col] - (*e)[row][col]) < 0.00001f))
                return false;
    return true;
}

std::shared_ptr<SceneCameraNode> SetCamera(TestScenes& scenes) {
    scenes.scene = std::make_shared<Scene>("billiard camera");
    auto camera = std::make_shared<SceneCameraNode>("camera");
    camera->MoveBy(0, -5, 2);
    scenes.scene->SceneGraph->AppendChild(camera);
    scenes.scene->CameraNodes.emplace("camera", camera);
    return camera;
}

void TestLoading() {
    TestScenes scenes;
    TestPhysics physics;
    BilliardGameLogic billiard(scenes, physics);
    IGameLogic& logic = billiard;
    Expect(logic.Initialize() == 0 &&
               scenes.loads == std::vector<std::string>{"Scene/billiard.ogex"},
           "loading uses the injected scene service without Application");
    logic.Tick();
    logic.Finalize();
    scenes.loadResult = -7;
    Expect(logic.Initialize() == -7 && scenes.loads.size() == 2 &&
               scenes.loads.back() == "Scene/billiard.ogex",
           "fresh initialization propagates the scene load failure unchanged");
    Expect(physics.forces.empty(), "loading does not apply forces");
}

void TestCueRouting() {
    TestScenes scenes;
    TestPhysics physics;
    BilliardGameLogic billiard(scenes, physics);
    IGameLogic& logic = billiard;
    logic.OnLeftKey();
    Expect(physics.forces.empty(), "absent cue does not reach physics");
    auto first = std::make_shared<SceneGeometryNode>("pbb_cue");
    scenes.cue = first;
    logic.OnLeftKey();
    auto replacement = std::make_shared<SceneGeometryNode>("pbb_cue");
    scenes.cue = replacement;
    logic.OnLeftKey();
    physics.hasBody = false;
    logic.OnLeftKey();
    Expect(physics.forces.size() == 3, "one force request per existing-cue event");
    if (physics.forces.size() == 3) {
        Expect(physics.forces[0].node == first.get() &&
                   physics.forces[1].node == replacement.get() &&
                   physics.forces[2].node == replacement.get(),
               "same-name replacement routes to the current node, including no-body result");
        for (const auto& force : physics.forces)
            Expect(force.value[0] == -100.0f && force.value[1] == 0.0f &&
                       force.value[2] == 0.0f,
                   "cue force preserves magnitude, direction and zero other components");
    }
    replacement.reset();
    Expect(scenes.cue.expired(), "logic does not retain a replaced cue");
    logic.OnLeftKey();
    Expect(physics.forces.size() == 3, "expired cue does not reach physics");
    Expect(scenes.geometryLookups == std::vector<std::string>(5, "pbb_cue"),
           "each cue event resolves the existing cue key through injected scenes");
}

void TestCameraRouting() {
    TestScenes scenes;
    TestPhysics physics;
    TestApplication app;
    BilliardGameLogic billiard(scenes, physics);
    IGameLogic& logic = billiard;
    logic.SetAppPointer(&app);
    auto first = SetCamera(scenes);
    SceneCameraNode expected("expected");
    expected.MoveBy(0, -5, 2);
    logic.OnAnalogStick(0, 100, -75);
    Expect(SameTransform(*first, expected), "left drag is ignored");
    logic.OnAnalogStick(1, 100, -75);
    expected.RotateBy(100.0f / 800 * PI, -75.0f / 600 * PI, 0);
    Expect(SameTransform(*first, expected),
           "right drag preserves signs and separate width/height scaling via IApplication");
    auto replacement = SetCamera(scenes);
    app.config.screenWidth = 1600;
    app.config.screenHeight = 900;
    logic.OnAnalogStick(1, -80, 90);
    SceneCameraNode expectedReplacement("expected replacement");
    expectedReplacement.MoveBy(0, -5, 2);
    expectedReplacement.RotateBy(-80.0f / 1600 * PI, 90.0f / 900 * PI, 0);
    Expect(SameTransform(*first, expected) &&
               SameTransform(*replacement, expectedReplacement),
           "new events use current scene camera and current window configuration");
    scenes.scene = std::make_shared<Scene>("no camera");
    logic.OnAnalogStick(1, 1, 1);
    scenes.scene.reset();
    logic.OnAnalogStick(1, 1, 1);
    Expect(physics.forces.empty(), "camera input does not apply physics force");
}
}  // namespace

int main() {
    TestLoading();
    TestCueRouting();
    TestCameraRouting();
    return failures ? 1 : 0;
}

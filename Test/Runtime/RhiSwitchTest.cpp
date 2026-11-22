#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "RhiSwitch.hpp"
#include "Framework/Manager/GraphicsManager.hpp"
#include "config.h"

#include "Framework/Interface/IAssetReader.hpp"
#include "Framework/Interface/ISceneManager.hpp"

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
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    int LoadScene(const char*) override { return 0; }
    uint64_t GetSceneRevision() const override { return 1; }
    const std::shared_ptr<Scene> GetSceneForRendering() const override { return {}; }
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const override { return {}; }
    void ResetScene() override {}
    std::weak_ptr<BaseSceneNode> GetRootNode() const override { return {}; }
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(const std::string&) const override { return {}; }
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(const std::string&) const override { return {}; }
};

class TestAssets : public IAssetReader {
   public:
    Buffer SyncOpenAndReadBinary(const char*) override { return {}; }
    std::string SyncOpenAndReadTextFileToString(const char*) override { return {}; }
};

class StubPipelineManager : public IPipelineStateManager {
   public:
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    bool RegisterPipelineState(PipelineState&) override { return true; }
    void UnregisterPipelineState(PipelineState&) override {}
    void Clear() override {}
    const std::shared_ptr<PipelineState> GetPipelineState(std::string) const override {
        return {};
    }
};

// The initial composition must use a backend compiled on this platform;
// switch targets use stub backends/managers, so they are platform-agnostic.
RhiBackend CompiledInitialBackend() {
#if defined(OS_WINDOWS)
    return RhiBackend::D3d12;
#elif defined(OS_MACOS)
    return RhiBackend::Metal;
#else
    return RhiBackend::OpenGL;
#endif
}

class TestApp : public BaseApplication {
   public:
    uint32_t fbWidth = 320;
    uint32_t fbHeight = 240;
    void GetFramebufferSize(uint32_t& width, uint32_t& height) override {
        width = fbWidth;
        height = fbHeight;
    }
};

class StubPlatform : public IPlatformHost {
   public:
    explicit StubPlatform(std::vector<std::string>& events) : m_Events(events) {}
    std::vector<RhiBackend> switchNotifications;
    void SetActiveBackend(IRenderBackend*) override {}
    int InitializeGuiPlatform() override { return 0; }
    void FinalizeGuiPlatform() override {}
    void FinalizePlatformResources() override { m_Events.push_back("platform-stop"); }
    void ProcessEvents() override {}
    void OnBackendSwitch(RhiBackend target) override {
        switchNotifications.push_back(target);
    }

   private:
    std::vector<std::string>& m_Events;
};

class StubBackend : public IRenderBackend {
   public:
    StubBackend(std::vector<std::string>& events, bool recreateWindow)
        : m_Events(events), m_RecreateWindow(recreateWindow) {}
    int graphicsResult = 0;
    void CreateMainWindow() override { m_Events.push_back("window-create"); }
    int InitializeGraphicsResources() override {
        m_Events.push_back("graphics-init");
        return graphicsResult;
    }
    void WaitForGraphicsIdle() override { m_Events.push_back("drain"); }
    void FinalizeGraphicsResources() override { m_Events.push_back("graphics-stop"); }
    int InitializeGuiPlatform() override {
        m_Events.push_back("gui-init");
        return 0;
    }
    void FinalizeGuiPlatform() override { m_Events.push_back("gui-stop"); }
    bool RequiresWindowRecreation() const override { return m_RecreateWindow; }

   private:
    std::vector<std::string>& m_Events;
    bool m_RecreateWindow;
};

std::vector<RhiBackendAvailability> AllReady() {
    return {{RhiBackend::D3d12, RhiAvailability::ReadyForInitialization, {}},
            {RhiBackend::OpenGL, RhiAvailability::ReadyForInitialization, {}},
            {RhiBackend::Metal, RhiAvailability::ReadyForInitialization, {}},
            {RhiBackend::Vulkan, RhiAvailability::ReadyForInitialization, {}}};
}

struct Fixture {
    std::vector<std::string> events;
    TestScenes scenes;
    TestAssets assets;
    GfxConfiguration config;
    TestApp app;
    StubPlatform platform{events};
    std::unique_ptr<RhiStartupComposition> composition;
    std::unique_ptr<RuntimeRoot> root;
    std::vector<RhiBackend> backendFailures;
    std::vector<RhiBackend> sessionFailures;
    bool failAllBackends = false;

    RhiBackendResult MakeBackend(RhiBackend target, BaseApplication&) {
        if (failAllBackends ||
            std::find(backendFailures.begin(), backendFailures.end(), target) !=
                backendFailures.end()) {
            return {nullptr, false, false, "injected backend failure"};
        }
        const bool recreate = target == RhiBackend::OpenGL;
        auto backend = std::make_unique<StubBackend>(events, recreate);
        return {std::move(backend), recreate, recreate, {}};
    }

    RhiSessionResult MakeSession(RhiBackend target, IRenderBackend& backend,
                                 ISceneManager& scenes, IAssetReader& assets,
                                 std::function<MemorySystem*()> provider) {
        if (std::find(sessionFailures.begin(), sessionFailures.end(), target) !=
            sessionFailures.end()) {
            return {{}, "injected session failure"};
        }
        // Headless stub managers: the base GraphicsManager initializes
        // without a window or GPU; no real backend cast is involved.
        (void)backend; (void)assets; (void)provider;
        RhiSessionModules modules;
        modules.pipelineStateManager = std::make_unique<StubPipelineManager>();
        modules.graphicsManager =
            std::make_unique<GraphicsManager>(scenes, *modules.pipelineStateManager);
        return {std::move(modules), {}};
    }

    explicit Fixture(RhiBackend initial = RhiBackend::Vulkan) {
        auto composed = CreateRhiStartupComposition(initial, config, scenes, assets);
        composition = std::move(composed.composition);
        root = std::make_unique<RuntimeRoot>();
        root->SetHost(app, platform, *composition->sessionBackend);
    }

    std::unique_ptr<RhiSwitchController> MakeController() {
        return std::make_unique<RhiSwitchController>(
            *root, *composition, config, scenes, assets,
            [this] { return root->GetMemorySystem(); },
            [this](RhiBackend t, BaseApplication& h) { return MakeBackend(t, h); },
            [this](RhiBackend t, IRenderBackend& b, ISceneManager& s,
                   IAssetReader& a, std::function<MemorySystem*()> p) {
                return MakeSession(t, b, s, a, p);
            },
            [] { return AllReady(); },
            [this](uint32_t& w, uint32_t& h) { w = app.fbWidth; h = app.fbHeight; });
    }
};

void TestRequestContract() {
    Fixture f;
    auto controller = f.MakeController();
    Expect(controller->Request(RhiBackend::Vulkan) ==
               RhiSwitchRequestResult::AlreadyActive,
           "selecting the active backend is a no-op");
    Expect(controller->Request(RhiBackend::OpenGL) ==
               RhiSwitchRequestResult::Accepted,
           "switch request accepted");
    Expect(controller->Request(RhiBackend::Metal) ==
               RhiSwitchRequestResult::Overlapping,
           "overlapping requests rejected while one is pending");
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::FirstFrame,
           "switch enters the first-frame gate");
    Expect(controller->IsSimulationPaused(), "simulation pauses during the gate");
    Expect(controller->Request(RhiBackend::Metal) ==
               RhiSwitchRequestResult::Overlapping,
           "overlapping requests rejected during the gate");
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::Running,
           "presented frame publishes success");
    Expect(!controller->IsSimulationPaused(), "simulation resumes");
}

void TestSuccessfulSwitchCycle() {
    Fixture f(CompiledInitialBackend());
    auto controller = f.MakeController();
    const auto generation0 = f.root->Session().Generation();
    const auto* graphics0 = f.composition->graphicsManager.get();
    Expect(f.config.fixOpenGLPerspectiveMatrix ==
               (CompiledInitialBackend() == RhiBackend::OpenGL),
           "initial convention follows the initial backend");

    Expect(controller->Request(RhiBackend::Vulkan) ==
               RhiSwitchRequestResult::Accepted, "request A->B");
    controller->ProcessFrameBoundary();
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::Running, "A->B completes");
    Expect(controller->ActiveBackend() == RhiBackend::Vulkan, "active backend is B");
    Expect(f.composition->graphicsManager.get() != graphics0,
           "session modules were replaced");
    Expect(f.root->Session().Generation() > generation0,
           "switch publishes new generations");
    Expect(!f.config.fixOpenGLPerspectiveMatrix,
           "render convention follows the target backend");
    Expect(controller->ConsumeFramePlanDirty(), "frame plan marked dirty");
    Expect(!controller->ConsumeFramePlanDirty(), "dirty flag consumed once");
    Expect(f.platform.switchNotifications.size() == 1 &&
               f.platform.switchNotifications[0] == RhiBackend::Vulkan,
           "host notified of the backend switch");

    Expect(controller->Request(RhiBackend::OpenGL) ==
               RhiSwitchRequestResult::Accepted, "request B->A");
    controller->ProcessFrameBoundary();
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::Running, "B->A completes");
    Expect(controller->ActiveBackend() == RhiBackend::OpenGL, "restored backend A");
    Expect(f.config.fixOpenGLPerspectiveMatrix ==
               (CompiledInitialBackend() == RhiBackend::OpenGL),
           "initial convention restored");
}

void TestZeroSizeGate() {
    Fixture f;
    auto controller = f.MakeController();
    controller->Request(RhiBackend::OpenGL);
    controller->ProcessFrameBoundary();
    f.app.fbWidth = 0;
    f.app.fbHeight = 0;
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::FirstFrame,
           "zero-size framebuffer does not publish success");
    f.app.fbWidth = 320;
    f.app.fbHeight = 240;
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::Running,
           "usable framebuffer publishes success");
}

void TestPreflightKeepsOldSession() {
    Fixture f;
    auto controller = f.MakeController();
    const auto* graphics0 = f.composition->graphicsManager.get();
    auto probe = [] {
        auto availability = AllReady();
        for (auto& a : availability) {
            if (a.backend == RhiBackend::Metal) {
                a.state = RhiAvailability::NotCompiled;
                a.reason = "Metal is a macOS backend";
            }
        }
        return availability;
    };
    RhiSwitchController withProbe(
        *f.root, *f.composition, f.config, f.scenes, f.assets,
        [&f] { return f.root->GetMemorySystem(); },
        [&f](RhiBackend t, BaseApplication& h) { return f.MakeBackend(t, h); },
        [&f](RhiBackend t, IRenderBackend& b, ISceneManager& s, IAssetReader& a,
             std::function<MemorySystem*()> p) {
            return f.MakeSession(t, b, s, a, p);
        },
        probe,
        [&f](uint32_t& w, uint32_t& h) { w = f.app.fbWidth; h = f.app.fbHeight; });
    Expect(withProbe.Request(RhiBackend::Metal) ==
               RhiSwitchRequestResult::NotCompiled,
           "uncompiled target fails preflight");
    withProbe.ProcessFrameBoundary();
    Expect(withProbe.State() == RhiSwitchState::Running,
           "preflight failure keeps the old session running");
    Expect(f.composition->graphicsManager.get() == graphics0,
           "preflight failure does not touch the session");
}

void TestRecoveryAfterRetirement() {
    Fixture f(CompiledInitialBackend());
    f.backendFailures.push_back(RhiBackend::Vulkan);
    auto controller = f.MakeController();
    controller->Request(RhiBackend::Vulkan);
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::FirstFrame,
           "recovered session enters the first-frame gate");
    Expect(controller->LastError().find("switch failed") != std::string::npos &&
               controller->LastError().find("recovered") != std::string::npos,
           "failed switch reported even when recovery succeeds");
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::Running, "recovery completes");
    Expect(controller->ActiveBackend() == CompiledInitialBackend(),
           "previous backend reconstructed");
}

void TestRecoveryFailureExits() {
    Fixture f;
    f.failAllBackends = true;
    auto controller = f.MakeController();
    controller->Request(RhiBackend::OpenGL);
    controller->ProcessFrameBoundary();
    Expect(controller->ExitRequested(),
           "recovery failure requests an orderly exit");
    Expect(controller->LastError().find("switch failed") != std::string::npos &&
               controller->LastError().find("recovery failed") != std::string::npos,
           "both errors reported");
    controller->ProcessFrameBoundary();
    Expect(controller->ExitRequested(), "exit state is terminal");
}

void TestSessionFailureRecovers() {
    Fixture f(CompiledInitialBackend());
    f.sessionFailures.push_back(RhiBackend::Vulkan);
    auto controller = f.MakeController();
    controller->Request(RhiBackend::Vulkan);
    controller->ProcessFrameBoundary();
    controller->ProcessFrameBoundary();
    Expect(controller->State() == RhiSwitchState::Running,
           "session creation failure recovers the previous backend");
    Expect(controller->ActiveBackend() == CompiledInitialBackend(),
           "previous backend active after session failure");
}
}  // namespace

int main() {
    TestRequestContract();
    TestSuccessfulSwitchCycle();
    TestZeroSizeGate();
    TestPreflightKeepsOldSession();
    TestRecoveryAfterRetirement();
    TestRecoveryFailureExits();
    TestSessionFailureRecovers();
    if (failures) return 1;
    std::cout << "RHI switch controller tests passed without windows or GPU\n";
    return 0;
}
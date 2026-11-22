#include "RhiComposition.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    uint64_t GetSceneRevision() const override { return 0; }
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

class CountingApplication : public BaseApplication {
   public:
    static int live;
    CountingApplication() { ++live; }
    ~CountingApplication() override { --live; }
};
int CountingApplication::live = 0;

class ThrowingPipelineManager : public IPipelineStateManager {
   public:
    ThrowingPipelineManager() {
        throw std::runtime_error("pipeline construction failed");
    }
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    bool RegisterPipelineState(PipelineState&) override { return false; }
    void UnregisterPipelineState(PipelineState&) override {}
    void Clear() override {}
    const std::shared_ptr<PipelineState> GetPipelineState(std::string) const override { return {}; }
};

std::vector<RhiBackend> CompiledBackends() {
    std::vector<RhiBackend> backends;
#if defined(OS_WINDOWS)
    backends.push_back(RhiBackend::D3d12);
#endif
#if defined(OS_WINDOWS) || defined(OS_LINUX)
    backends.push_back(RhiBackend::OpenGL);
#endif
#if defined(OS_MACOS)
    backends.push_back(RhiBackend::Metal);
#endif
#if defined(HAS_VULKAN)
    backends.push_back(RhiBackend::Vulkan);
#endif
    return backends;
}

bool IsCompiled(RhiBackend backend) {
    switch (backend) {
        case RhiBackend::D3d12:
#if defined(OS_WINDOWS)
            return true;
#else
            return false;
#endif
        case RhiBackend::OpenGL:
#if defined(OS_WINDOWS) || defined(OS_LINUX)
            return true;
#else
            return false;
#endif
        case RhiBackend::Metal:
#if defined(OS_MACOS)
            return true;
#else
            return false;
#endif
        case RhiBackend::Vulkan:
#if defined(HAS_VULKAN)
            return true;
#else
            return false;
#endif
    }
    return false;
}

void TestCompiledCompositions() {
    TestScenes scenes;
    TestAssets assets;
    GfxConfiguration config;
    const auto compiled = CompiledBackends();
    Expect(!compiled.empty(), "desktop build includes at least one backend");
    for (int cycle = 0; cycle < 3; ++cycle) {
        for (auto backend : compiled) {
            config.fixOpenGLPerspectiveMatrix = false;
            auto result = CreateRhiStartupComposition(backend, config, scenes, assets);
            Expect(result.composition != nullptr && result.error.empty(),
                   "compiled backend composes without initializing graphics");
            if (!result.composition) continue;
            Expect(result.composition->backend == backend,
                   "composition retains the selected backend");
            Expect(result.composition->application &&
                       result.composition->pipelineStateManager &&
                       result.composition->graphicsManager,
                   "composition owns application, pipeline, and graphics");
            const bool opengl = backend == RhiBackend::OpenGL;
            Expect(result.composition->fixOpenGLPerspectiveMatrix == opengl,
                   "render convention follows the active backend");
            Expect(config.fixOpenGLPerspectiveMatrix == opengl,
                   "application configuration carries the active convention");
        }
    }
}

void TestSessionModules() {
    TestScenes scenes;
    TestAssets assets;
    GfxConfiguration config;
    for (auto backend : CompiledBackends()) {
        auto startup = CreateRhiStartupComposition(backend, config, scenes, assets);
        Expect(startup.composition != nullptr, "host composition for session test");
        if (!startup.composition) continue;
        Expect(startup.composition->recreateWindowOnReconstruction ==
                   (backend == RhiBackend::OpenGL),
               "window recreation policy follows backend constraints");
        for (int cycle = 0; cycle < 3; ++cycle) {
            auto session = CreateRhiSessionModules(
                backend, *startup.composition->application, scenes, assets);
            Expect(session.error.empty() &&
                       session.modules.pipelineStateManager &&
                       session.modules.graphicsManager,
                   "session modules compose window/GPU-free on the host");
        }
    }
    for (auto backend : {RhiBackend::D3d12, RhiBackend::OpenGL,
                         RhiBackend::Metal, RhiBackend::Vulkan}) {
        if (IsCompiled(backend)) continue;
        CountingApplication host;
        auto session = CreateRhiSessionModules(backend, host, scenes, assets);
        Expect(!session.modules.graphicsManager &&
                   session.error.find("not compiled") != std::string::npos,
               "absent backend session fails explicitly");
    }
}

void TestAbsentBackends() {
    TestScenes scenes;
    TestAssets assets;
    GfxConfiguration config;
    for (auto backend : {RhiBackend::D3d12, RhiBackend::OpenGL,
                         RhiBackend::Metal, RhiBackend::Vulkan}) {
        if (IsCompiled(backend)) continue;
        auto result = CreateRhiStartupComposition(backend, config, scenes, assets);
        Expect(!result.composition &&
                   result.error.find("not compiled") != std::string::npos,
               "absent backend fails explicitly without fallback");
    }
    auto invalid = CreateRhiStartupComposition(
        static_cast<RhiBackend>(-1), config, scenes, assets);
    Expect(!invalid.composition && !invalid.error.empty(),
           "invalid backend value fails explicitly");
}

void TestFailedConstructionCleanup() {
    const int liveBefore = CountingApplication::live;
    try {
        auto composition = std::make_unique<RhiStartupComposition>();
        composition->application = std::make_unique<CountingApplication>();
        composition->pipelineStateManager =
            std::make_unique<ThrowingPipelineManager>();
        Expect(false, "throwing member construction must propagate");
    } catch (const std::runtime_error&) {
    }
    Expect(CountingApplication::live == liveBefore,
           "failed composition releases the already-owned application");
}

void TestAvailability() {
    const auto availability = ProbeRhiStartupAvailability();
    Expect(availability.size() == 4, "one availability entry per canonical backend");
    for (auto backend : {RhiBackend::D3d12, RhiBackend::OpenGL,
                         RhiBackend::Metal, RhiBackend::Vulkan}) {
        const RhiBackendAvailability* entry = nullptr;
        int count = 0;
        for (const auto& candidate : availability) {
            if (candidate.backend == backend) {
                entry = &candidate;
                ++count;
            }
        }
        Expect(count == 1 && entry, "backend appears exactly once");
        if (!entry) continue;
        if (!IsCompiled(backend)) {
            Expect(entry->state == RhiAvailability::NotCompiled &&
                       !entry->reason.empty(),
                   "absent backend reports not compiled with a reason");
        } else if (backend == RhiBackend::Vulkan) {
            Expect(entry->state == RhiAvailability::ReadyForInitialization ||
                       (entry->state == RhiAvailability::RuntimeUnavailable &&
                        !entry->reason.empty()),
                   "Vulkan availability carries loader evidence, not a device claim");
        } else {
            Expect(entry->state == RhiAvailability::ReadyForInitialization,
                   "included native backend is ready to attempt initialization");
        }
    }
#if defined(OS_WINDOWS) || defined(OS_LINUX)
    const auto selection =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::OpenGL, availability);
    Expect(selection.backend == RhiBackend::OpenGL && selection.error.empty(),
           "probed availability resolves the application default");
#endif
}
void TestTitleLabels() {
    const auto platform = RhiStartupPlatformName();
#if defined(OS_WINDOWS)
    Expect(platform == "Windows", "platform label matches the build platform");
#elif defined(OS_MACOS)
    Expect(platform == "macOS", "platform label matches the build platform");
#elif defined(OS_LINUX)
    Expect(platform == "Linux", "platform label matches the build platform");
#endif
    for (auto backend : CompiledBackends()) {
        const auto host = RhiStartupHostName(backend);
        Expect(!host.empty(), "every compiled backend has a host label");
#if defined(OS_LINUX)
        if (backend == RhiBackend::OpenGL) {
#if defined(HAS_SDL2)
            Expect(host == "SDL", "Linux OpenGL retains the SDL host");
#else
            Expect(host == "XCB", "Linux OpenGL falls back to the XCB host");
#endif
        } else {
            Expect(host == "XCB", "Linux Vulkan uses the XCB host");
        }
#endif
    }
}
}  // namespace

int main() {
    TestCompiledCompositions();
    TestSessionModules();
    TestAbsentBackends();
    TestFailedConstructionCleanup();
    TestAvailability();
    TestTitleLabels();
    if (failures) return 1;
    std::cout << "RHI composition tests passed without windows or GPU\n";
    return 0;
}
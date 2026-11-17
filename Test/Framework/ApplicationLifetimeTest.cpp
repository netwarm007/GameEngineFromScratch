#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "BaseApplication.hpp"
#include "MemoryManager.hpp"
#include "EngineAllocation.hpp"
#include "MemorySystem.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class RecordingApplication : public BaseApplication {
   public:
    void CreateMainWindow() override {
        windowAlive = true;
    }

    std::vector<std::string> events;
    bool windowAlive = false;
    bool graphicsAlive = false;
    bool gpuWorkPending = false;
    int graphicsResult = 0;

   protected:
    int InitializeGraphicsResources() override {
        if (!windowAlive) return -1;
        events.push_back("init graphics");
        graphicsAlive = true;
        return graphicsResult;
    }
    void WaitForGraphicsIdle() override {
        if (!graphicsAlive) return;
        Expect(windowAlive, "GPU synchronization runs while the window exists");
        events.push_back("wait graphics");
        gpuWorkPending = false;
    }
    void FinalizeGraphicsResources() override {
        if (!graphicsAlive) return;
        Expect(windowAlive, "backend cleanup runs while its native window exists");
        events.push_back("stop graphics");
        graphicsAlive = false;
    }
    void FinalizePlatformResources() override {
        if (!windowAlive) return;
        Expect(!graphicsAlive, "platform cleanup follows backend cleanup");
        events.push_back("stop window");
        windowAlive = false;
    }
};

class RecordingModule : public IMemoryManager {
   public:
    RecordingModule(std::string name, RecordingApplication& app)
        : m_Name(std::move(name)), m_App(app) {}

    int Initialize() override {
        Expect(m_App.windowAlive && m_App.graphicsAlive,
               "module initialization sees live platform and backend resources");
        m_App.events.push_back("init " + m_Name);
        if (submitWorkOnInitialize) m_App.gpuWorkPending = true;
        return result;
    }
    void Finalize() override {
        Expect(m_App.windowAlive && m_App.graphicsAlive,
               "module cleanup and rollback precede backend and window teardown");
        Expect(!m_App.gpuWorkPending,
               "GPU work completes before any module releases referenced resources");
        m_App.events.push_back("stop " + m_Name);
    }
    void Tick() override {
        m_App.gpuWorkPending = true;
        m_App.events.push_back("tick " + m_Name);
    }
    void* AllocatePage(size_t) override { return nullptr; }
    void FreePage(void*) override {}

    int result = 0;
    bool submitWorkOnInitialize = false;

   private:
    std::string m_Name;
    RecordingApplication& m_App;
};

void TestSuccessfulShutdown() {
    RecordingApplication app;
    RecordingModule first("first", app), second("second", app);
    app.CreateMainWindow();
    Expect(!app.graphicsAlive, "window creation does not initialize graphics");
    app.RegisterManagerModule(&first);
    app.RegisterManagerModule(&second);
    Expect(app.Initialize() == 0, "application initialization succeeds");
    app.Tick();
    BaseApplication& application = app;
    application.Finalize();
    Expect(app.events == std::vector<std::string>{
               "init graphics", "init first", "init second",
               "tick first", "tick second",
               "wait graphics", "stop second", "stop first", "stop graphics",
               "stop window"},
           "application synchronizes GPU work before reverse module, backend, and platform shutdown");
    Expect(!app.graphicsAlive && !app.windowAlive,
           "application shutdown releases both outer resource layers");
    app.events.clear();
    app.Tick();
    Expect(app.events.empty(), "shutdown leaves no modules in the frame loop");
    app.CreateMainWindow();
    app.RegisterManagerModule(&first);
    Expect(app.Initialize() == 0, "a new lifecycle initializes graphics again");
    Expect(app.events == std::vector<std::string>{"init graphics", "init first"},
           "finalization resets successful backend initialization state");
    app.Finalize();
}

void TestFailedModuleInitialization() {
    for (int failingIndex = 0; failingIndex < 3; ++failingIndex) {
        RecordingApplication app;
        RecordingModule first("first", app), second("second", app),
            third("third", app);
        RecordingModule* ordered[] = {&first, &second, &third};
        const std::string names[] = {"first", "second", "third"};
        app.CreateMainWindow();
        for (auto* module : ordered) {
            module->submitWorkOnInitialize = true;
            app.RegisterManagerModule(module);
        }
        ordered[failingIndex]->result = 17;
        Expect(app.Initialize() == 17, "application preserves the module error");
        std::vector<std::string> expected{"init graphics"};
        for (int i = 0; i <= failingIndex; ++i)
            expected.push_back("init " + names[i]);
        expected.push_back("wait graphics");
        for (int i = failingIndex - 1; i >= 0; --i)
            expected.push_back("stop " + names[i]);
        Expect(!app.gpuWorkPending,
               "even first-module failure drains initialization GPU work");
        Expect(app.events == expected && app.graphicsAlive && app.windowAlive,
               "rollback retains platform resources until application shutdown");
        app.Tick();
        Expect(app.events == expected, "failed modules do not enter the frame loop");
        app.Finalize();
        expected.push_back("wait graphics");
        expected.push_back("stop graphics");
        expected.push_back("stop window");
        Expect(app.events == expected && !app.graphicsAlive && !app.windowAlive,
               "failed initialization shutdown releases outer resources without re-finalizing modules");
    }
}

void TestRetryAfterGpuRollback() {
    RecordingApplication app;
    RecordingModule first("first", app), second("second", app);
    first.submitWorkOnInitialize = true;
    second.submitWorkOnInitialize = true;
    second.result = 17;
    app.CreateMainWindow();
    app.RegisterManagerModule(&first);
    app.RegisterManagerModule(&second);
    Expect(app.Initialize() == 17, "failed GPU initialization can be retried");
    Expect(app.events == std::vector<std::string>{
               "init graphics", "init first", "init second",
               "wait graphics", "stop first"},
           "rollback synchronization precedes any initialized module cleanup");
    app.events.clear();
    second.result = 0;
    Expect(app.Initialize() == 0, "retry retains registrations");
    Expect(app.events == std::vector<std::string>{"init first", "init second"} &&
               app.gpuWorkPending,
           "successful initialization does not invoke rollback synchronization");
    app.events.clear();
    Expect(app.Initialize() != 0, "running application rejects initialization");
    Expect(app.events.empty() && app.gpuWorkPending,
           "rejected initialization does not synchronize or unwind live modules");
    app.Finalize();
    Expect(app.events == std::vector<std::string>{
               "wait graphics", "stop second", "stop first", "stop graphics",
               "stop window"},
           "retry finalizes initialized modules exactly once after synchronization");
}

void TestWindowWithoutModules() {
    RecordingApplication app;
    app.CreateMainWindow();
    app.Finalize();
    Expect(app.events == std::vector<std::string>{"stop window"},
           "window-only clients do not create backend resources for cleanup");
}

void TestBackendInitializationBoundaries() {
    RecordingApplication app;
    RecordingModule module("module", app);
    app.RegisterManagerModule(&module);
    Expect(app.Initialize() == -1 && app.events.empty(),
           "missing window prevents backend and module initialization");
    app.CreateMainWindow();
    app.graphicsResult = 23;
    Expect(app.Initialize() == 23,
           "application preserves the backend initialization error");
    app.Tick();
    Expect(app.events == std::vector<std::string>{"init graphics"},
           "backend failure prevents module initialization and ticks");
    app.Finalize();
    Expect(app.events == std::vector<std::string>{
               "init graphics", "wait graphics", "stop graphics", "stop window"},
           "explicit shutdown cleans partial backend resources before the window");
    app.events.clear();
    app.graphicsResult = 0;
    app.CreateMainWindow();
    app.RegisterManagerModule(&module);
    Expect(app.Initialize() == 0,
           "a new lifecycle initializes the backend after prior failure cleanup");
    app.Finalize();
    Expect(app.events == std::vector<std::string>{
               "init graphics", "init module", "wait graphics", "stop module",
               "stop graphics", "stop window"},
           "backend failure does not poison a subsequent lifecycle");
}

class GuiRecordingApplication : public RecordingApplication {
   public:
    int guiResult = 0;
    bool guiAlive = false;

   protected:
    int InitializeGuiPlatform() override {
        Expect(windowAlive && graphicsAlive,
               "GUI input backend binds after the final native view exists");
        events.push_back("init gui");
        guiAlive = guiResult == 0;
        return guiResult;
    }
    void FinalizeGuiPlatform() override {
        Expect(windowAlive && graphicsAlive && guiAlive && !gpuWorkPending,
               "GUI shutdown runs once with idle graphics and a live native view");
        events.push_back("stop gui");
        guiAlive = false;
    }
};

void TestGuiPlatformLifetime() {
    GuiRecordingApplication app;
    RecordingModule module("renderer", app), failing("later", app);
    app.CreateMainWindow();
    app.RegisterManagerModule(&module);
    app.RegisterManagerModule(&failing);
    app.guiResult = 29;
    Expect(app.Initialize() == 29, "GUI platform errors propagate before modules");
    Expect(app.events == std::vector<std::string>{"init graphics", "init gui"},
           "GUI platform failure does not initialize a renderer or modules");
    app.events.clear();
    app.guiResult = 0;
    failing.result = 17;
    Expect(app.Initialize() == 17, "module failure after GUI setup propagates");
    Expect(app.events == std::vector<std::string>{
               "init gui", "init renderer", "init later",
               "wait graphics", "stop renderer"} && app.guiAlive,
           "renderer rollback retains GUI input backend for retry");
    app.events.clear();
    failing.result = 0;
    Expect(app.Initialize() == 0, "renderer can retry with the existing GUI platform");
    Expect(app.events == std::vector<std::string>{"init renderer", "init later"},
           "retry does not initialize GUI input backend twice");
    app.events.clear();
    app.Finalize();
    app.Finalize();
    Expect(app.events == std::vector<std::string>{
               "wait graphics", "stop later", "stop renderer", "stop gui",
               "stop graphics", "stop window"},
           "modules, GUI input, graphics and native window shut down in order once");

    GuiRecordingApplication failed;
    failed.CreateMainWindow();
    failed.guiResult = 29;
    Expect(failed.Initialize() == 29, "failed GUI setup can be finalized");
    failed.events.clear();
    failed.Finalize();
    Expect(failed.events == std::vector<std::string>{
               "wait graphics", "stop graphics", "stop window"},
           "failed GUI initialization is not passed to its shutdown hook");
}
}  // namespace

#ifdef USE_ENGINE_ALLOCATORS
// Records whether the memory system still exists when a module finalizes.
class AllocatingModule : public RecordingModule {
   public:
    AllocatingModule(RecordingApplication& app) : RecordingModule("alloc", app),
        m_App(app) {}
    int Initialize() override {
        m_Object = MakeEngineShared<int>(42);
        return RecordingModule::Initialize();
    }
    void Finalize() override {
        sawSystemAtFinalize = m_App.GetMemorySystem() != nullptr;
        m_Object.reset();
        RecordingModule::Finalize();
    }
    bool sawSystemAtFinalize = false;

   private:
    RecordingApplication& m_App;
    std::shared_ptr<int> m_Object;
};

void TestMemorySystemLifetime() {
    {
        RecordingApplication app;
        AllocatingModule module(app);
        app.CreateMainWindow();
        app.RegisterManagerModule(static_cast<IRuntimeModule*>(&module));
        Expect(app.GetMemorySystem() == nullptr,
               "no memory system before initialization");
        Expect(app.Initialize() == 0, "application starts");
        Expect(app.GetMemorySystem() && app.GetMemorySystem()->IsConfigured(),
               "memory system exists while running");
        Expect(EngineAllocationCount() >= 1,
               "module objects come from the engine allocator");
        const auto frame = app.GetMemorySystem()->Frame().FrameNumber();
        app.Tick();
        Expect(app.GetMemorySystem()->Frame().FrameNumber() == frame + 1,
               "each application tick begins a new frame arena");
        Expect(app.Initialize() != 0 && app.GetMemorySystem() != nullptr,
               "rejected re-initialization keeps the memory system");
        app.Finalize();
        Expect(module.sawSystemAtFinalize,
               "memory system outlives module finalization");
        Expect(app.GetMemorySystem() == nullptr,
               "memory system is released after shutdown");
    }
    {
        RecordingApplication app;
        RecordingModule failing("failing", app);
        failing.result = 17;
        app.CreateMainWindow();
        app.RegisterManagerModule(static_cast<IRuntimeModule*>(&failing));
        Expect(app.Initialize() == 17, "module failure propagates");
        Expect(app.GetMemorySystem() == nullptr,
               "failed initialization releases the memory system");
        app.Finalize();
    }
}
#endif

int main() {
    TestSuccessfulShutdown();
    TestFailedModuleInitialization();
    TestRetryAfterGpuRollback();
    TestWindowWithoutModules();
    TestBackendInitializationBoundaries();
    TestGuiPlatformLifetime();
#ifdef USE_ENGINE_ALLOCATORS
    TestMemorySystemLifetime();
#endif
    return failures == 0 ? 0 : 1;
}

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "BaseApplication.hpp"

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
        graphicsAlive = true;
    }

    std::vector<std::string> events;
    bool windowAlive = false;
    bool graphicsAlive = false;
    bool gpuWorkPending = false;

   protected:
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

   private:
    std::string m_Name;
    RecordingApplication& m_App;
};

void TestSuccessfulShutdown() {
    RecordingApplication app;
    RecordingModule first("first", app), second("second", app);
    app.CreateMainWindow();
    app.RegisterManagerModule(&first);
    app.RegisterManagerModule(&second);
    Expect(app.Initialize() == 0, "application initialization succeeds");
    app.Tick();
    BaseApplication& application = app;
    application.Finalize();
    Expect(app.events == std::vector<std::string>{
               "init first", "init second", "tick first", "tick second",
               "wait graphics", "stop second", "stop first", "stop graphics",
               "stop window"},
           "application synchronizes GPU work before reverse module, backend, and platform shutdown");
    Expect(!app.graphicsAlive && !app.windowAlive,
           "application shutdown releases both outer resource layers");
    app.events.clear();
    app.Tick();
    Expect(app.events.empty(), "shutdown leaves no modules in the frame loop");
}

void TestFailedModuleInitialization() {
    for (int failingIndex = 0; failingIndex < 3; ++failingIndex) {
        RecordingApplication app;
        RecordingModule first("first", app), second("second", app),
            third("third", app);
        RecordingModule* ordered[] = {&first, &second, &third};
        const std::string names[] = {"first", "second", "third"};
        app.CreateMainWindow();
        for (auto* module : ordered) app.RegisterManagerModule(module);
        ordered[failingIndex]->result = 17;
        Expect(app.Initialize() == 17, "application preserves the module error");
        std::vector<std::string> expected;
        for (int i = 0; i <= failingIndex; ++i)
            expected.push_back("init " + names[i]);
        for (int i = failingIndex - 1; i >= 0; --i)
            expected.push_back("stop " + names[i]);
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

void TestWindowWithoutModules() {
    RecordingApplication app;
    app.CreateMainWindow();
    app.Finalize();
    Expect(app.events ==
               std::vector<std::string>{"wait graphics", "stop graphics",
                                        "stop window"},
           "window-only clients release resources without initializing modules");
}
}  // namespace

int main() {
    TestSuccessfulShutdown();
    TestFailedModuleInitialization();
    TestWindowWithoutModules();
    return failures == 0 ? 0 : 1;
}

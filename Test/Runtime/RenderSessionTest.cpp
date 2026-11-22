#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "EngineAllocation.hpp"
#include "FrameScheduler.hpp"
#include "RenderSession.hpp"
#include "RuntimeRoot.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class RecordingModule : public IRuntimeModule {
   public:
    RecordingModule(std::string name, std::vector<std::string>& events)
        : m_Name(std::move(name)), m_Events(events) {}

    int Initialize() override {
        m_Events.push_back("init " + m_Name);
        return result;
    }
    void Finalize() override { m_Events.push_back("stop " + m_Name); }
    void Tick() override { m_Events.push_back("tick " + m_Name); }

    int result = 0;

   private:
    std::string m_Name;
    std::vector<std::string>& m_Events;
};

void TestPersistentUnaffectedBySessionCycles() {
    std::vector<std::string> events;
    RecordingModule persistent("persistent", events);
    RecordingModule sessionA("sessionA", events), sessionB("sessionB", events);
    RuntimeRoot root;
    root.PersistentModules().Register(persistent);
    Expect(root.PersistentModules().Initialize() == 0,
           "persistent services initialize once");
    events.clear();

    for (int cycle = 0; cycle < 3; ++cycle) {
        root.Session().Register(sessionA);
        root.Session().Register(sessionB);
        Expect(root.Session().Activate() == 0, "session activation succeeds");
        Expect(root.Session().IsActive(), "session is active");
        root.Session().Retire();
        Expect(!root.Session().IsActive(), "session retired");
    }
    Expect(events == std::vector<std::string>{
                         "init sessionA", "init sessionB",
                         "stop sessionB", "stop sessionA",
                         "init sessionA", "init sessionB",
                         "stop sessionB", "stop sessionA",
                         "init sessionA", "init sessionB",
                         "stop sessionB", "stop sessionA"},
           "repeated session cycles never touch persistent services");
    root.PersistentModules().Finalize();
    Expect(events.back() == "stop persistent",
           "persistent services finalize only on explicit shutdown");
}

void TestSessionFailureRollbackAndRetry() {
    std::vector<std::string> events;
    RecordingModule persistent("persistent", events);
    RecordingModule first("first", events), second("second", events);
    RuntimeRoot root;
    root.PersistentModules().Register(persistent);
    Expect(root.PersistentModules().Initialize() == 0, "persistent up");
    events.clear();

    root.Session().Register(first);
    root.Session().Register(second);
    second.result = 23;
    bool drained = false;
    const int result = root.Session().Activate([&] {
        drained = true;
        events.push_back("drain");
    });
    Expect(result == 23, "session activation preserves the module error");
    Expect(drained, "owner drains GPU work before rollback");
    Expect(events == std::vector<std::string>{"init first", "init second",
                                              "drain", "stop first"},
           "rollback finalizes only initialized session modules, in reverse");
    Expect(!root.Session().IsActive(), "failed activation leaves no session");
    Expect(root.Session().GetSessionModules().empty(),
           "failed session has no frame participants");
    Expect(root.PersistentModules().GetInitializedModules().size() == 1,
           "persistent services survive session failure");

    second.result = 0;
    Expect(root.Session().Activate() == 0, "retry after rollback succeeds");
    Expect(root.Session().IsActive(), "retried session is active");
    root.Session().Retire();
    root.PersistentModules().Finalize();
}

void TestGenerationSemantics() {
    std::vector<std::string> events;
    RecordingModule module("module", events);
    RenderSessionSlot slot;
    Expect(slot.Generation() == 0, "generation starts at zero");
    Expect(!slot.IsActive(), "no session before activation");

    slot.Register(module);
    Expect(slot.Activate() == 0, "first activation");
    const auto first = slot.Generation();
    Expect(first == 1, "activation publishes a new generation");

    slot.Retire();
    const auto gap = slot.Generation();
    Expect(gap == 2, "retirement invalidates borrowed references immediately");
    slot.Retire();
    Expect(slot.Generation() == gap, "repeated retirement is a no-op");

    slot.Register(module);
    Expect(slot.Activate() == 0, "recreation after retirement");
    Expect(slot.Generation() == 3, "recreated session has a fresh identity");

    bool rejected = false;
    try {
        slot.Register(module);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    Expect(rejected, "reject registration while a session is active");
    Expect(slot.Activate() != 0, "reject repeated activation");
    Expect(slot.Generation() == 3, "rejected operations keep the generation");

    slot.Retire();
    module.result = 9;
    slot.Register(module);
    Expect(slot.Activate() == 9, "failing recreation");
    Expect(slot.Generation() == 4,
           "failed recreation does not publish a session identity");
    module.result = 0;
    Expect(slot.Activate() == 0, "recovery after failed recreation");
    Expect(slot.Generation() == 5, "recovered session identity is fresh");
    slot.Retire();
    Expect(slot.Generation() == 6, "generation is monotonic across cycles");
}

void TestFrameOrderPreserved() {
    std::vector<std::string> events;
    RecordingModule p1("p1", events), p2("p2", events);
    RecordingModule s1("s1", events), s2("s2", events);
    RuntimeRoot root;
    root.PersistentModules().Register(p1);
    root.PersistentModules().Register(p2);
    Expect(root.PersistentModules().Initialize() == 0, "persistent up");
    root.Session().Register(s1);
    root.Session().Register(s2);
    Expect(root.Session().Activate() == 0, "session up");
    events.clear();

    auto sessionModules = root.Session().GetSessionModules();
    Expect(sessionModules.size() == 2 && sessionModules[0] == &s1 &&
               sessionModules[1] == &s2,
           "session participants keep registration order");

    // The caller composes the interleaved frame order; the session boundary
    // must not force a persistent-then-session grouping.
    const std::array<IRuntimeModule*, 4> frame = {&p1, &s1, &p2, &s2};
    TickFrame(frame);
    Expect(events == std::vector<std::string>{"tick p1", "tick s1", "tick p2",
                                              "tick s2"},
           "caller-composed interleaved frame order is preserved");
    events.clear();

    root.Session().Retire();
    RecordingModule s3("s3", events);
    root.Session().Register(s3);
    Expect(root.Session().Activate() == 0, "new session participants");
    events.clear();
    const std::array<IRuntimeModule*, 3> nextFrame = {&p1, &s3, &p2};
    TickFrame(nextFrame);
    Expect(events == std::vector<std::string>{"tick p1", "tick s3", "tick p2"},
           "reconstruction swaps participants without reordering the frame");
    root.Session().Retire();
    root.PersistentModules().Finalize();
}

class StubHost : public IApplication, public IPlatformHost, public IRenderBackend {
   public:
    explicit StubHost(std::vector<std::string>& events) : m_Events(events) {}

    int graphicsResult = 0;
    int guiResult = 0;

    // IRenderBackend
    int InitializeGraphicsResources() override {
        m_Events.push_back("graphics-init");
        return graphicsResult;
    }
    void WaitForGraphicsIdle() override { m_Events.push_back("drain"); }
    void FinalizeGraphicsResources() override {
        m_Events.push_back("graphics-stop");
    }

    // IPlatformHost
    int InitializeGuiPlatform() override {
        m_Events.push_back("gui-init");
        return guiResult;
    }
    void FinalizeGuiPlatform() override { m_Events.push_back("gui-stop"); }
    void FinalizePlatformResources() override {
        m_Events.push_back("platform-stop");
    }
    void ProcessEvents() override { m_Events.push_back("events"); }

    // IApplication
    void SetCommandLineParameters(int, char**) override {}
    int GetCommandLineArgumentsCount() const override { return 0; }
    const char* GetCommandLineArgument(int) const override { return nullptr; }
    bool IsQuit() const override { return false; }
    void RequestQuit() override {}
    void CreateMainWindow() override { m_Events.push_back("window-create"); }
    void* GetMainWindowHandler() override { return nullptr; }
    const GfxConfiguration GetConfiguration() const override {
        return GfxConfiguration();
    }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) override {
        width = 320;
        height = 240;
    }

   private:
    std::vector<std::string>& m_Events;
};

void TestRootStartupAndTeardownOrder() {
    std::vector<std::string> events;
    StubHost host(events);
    RecordingModule p1("p1", events), p2("p2", events);
    RecordingModule s1("s1", events), s2("s2", events);
    RuntimeRoot root;
    root.SetHost(host, host, host);
    root.RegisterPersistentModule(p1);
    root.RegisterPersistentModule(p2);
    root.RegisterSessionModule(s1);
    root.RegisterSessionModule(s2);
    root.SetFramePlan({&p1, &s1, &p2, &s2});

    Expect(root.Initialize() == 0, "root startup succeeds");
    Expect(events == std::vector<std::string>{
                         "graphics-init", "gui-init", "init p1", "init p2",
                         "init s1", "init s2"},
           "startup order: backend, GUI platform, persistent, session");
    events.clear();

    root.Tick();
    Expect(events == std::vector<std::string>{"events", "tick p1", "tick s1",
                                              "tick p2", "tick s2"},
           "frame order: platform events then the caller-composed plan");
    events.clear();

    root.Finalize();
    Expect(events == std::vector<std::string>{
                         "drain", "stop s2", "stop s1", "stop p2", "stop p1",
                         "gui-stop", "graphics-stop", "platform-stop"},
           "teardown order: drain, session, persistent, GUI, backend, platform");
    events.clear();
    root.Finalize();
    Expect(events == std::vector<std::string>{"drain", "platform-stop"},
           "repeated Finalize does not double-finalize modules or hooks");
}

void TestRootFailurePaths() {
    {
        std::vector<std::string> events;
        StubHost host(events);
        host.graphicsResult = 5;
        RecordingModule p1("p1", events), s1("s1", events);
        RuntimeRoot root;
        root.SetHost(host, host, host);
        root.RegisterPersistentModule(p1);
        root.RegisterSessionModule(s1);
        Expect(root.Initialize() == 5, "graphics failure propagates");
        Expect(events == std::vector<std::string>{"graphics-init"},
               "graphics failure initializes no modules");
        events.clear();
        root.Finalize();
        Expect(events == std::vector<std::string>{"drain", "platform-stop"},
               "Finalize after graphics failure releases only platform");
    }
    {
        std::vector<std::string> events;
        StubHost host(events);
        RecordingModule p1("p1", events), p2("p2", events);
        RecordingModule s1("s1", events), s2("s2", events);
        s2.result = 7;
        RuntimeRoot root;
        root.SetHost(host, host, host);
        root.RegisterPersistentModule(p1);
        root.RegisterPersistentModule(p2);
        root.RegisterSessionModule(s1);
        root.RegisterSessionModule(s2);
        Expect(root.Initialize() == 7, "session failure propagates");
        Expect(events == std::vector<std::string>{
                             "graphics-init", "gui-init", "init p1", "init p2",
                             "init s1", "init s2", "drain", "stop s1",
                             "stop p2", "stop p1"},
               "session failure rolls back session and persistent modules");
        events.clear();
        root.Finalize();
        Expect(events == std::vector<std::string>{
                     "drain", "gui-stop", "graphics-stop", "platform-stop"},
               "Finalize after session failure releases native hooks once");
    }
}

void TestRootReconstructionCycle() {
    std::vector<std::string> events;
    StubHost host(events);
    RecordingModule p1("p1", events), p2("p2", events);
    RecordingModule s1("s1", events), s2("s2", events);
    RuntimeRoot root;
    root.SetHost(host, host, host);
    root.RegisterPersistentModule(p1);
    root.RegisterPersistentModule(p2);
    root.RegisterSessionModule(s1);
    root.RegisterSessionModule(s2);
    root.SetFramePlan({&p1, &s1, &p2, &s2});
    Expect(root.Initialize() == 0, "startup before reconstruction");
    const auto generation = root.Session().Generation();
    events.clear();

    // Same-backend reconstruction keeping the native window: the GUI
    // platform binding survives (it belongs to the window, not the session).
    root.RetireSessionForReconstruction(false);
    Expect(events == std::vector<std::string>{"drain", "stop s2", "stop s1",
                                              "graphics-stop"},
           "retirement releases session and backend state, not window/GUI");
    events.clear();
    Expect(root.PrepareReconstructedSession(false) == 0,
           "native backend recreates on the existing window");
    Expect(events == std::vector<std::string>{"graphics-init"},
           "preparation re-initializes only the backend on a kept window");
    events.clear();

    RecordingModule s3("s3", events);
    root.RegisterSessionModule(s3);
    Expect(root.Session().Activate(
               [&] { host.WaitForGraphicsIdle(); }) == 0,
           "fresh session modules activate");
    Expect(root.Session().Generation() > generation,
           "reconstruction publishes a new session generation");
    root.SetFramePlan({&p1, &s3, &p2});
    events.clear();
    root.Tick();
    Expect(events == std::vector<std::string>{"events", "tick p1", "tick s3",
                                              "tick p2"},
           "persistent services keep ticking across reconstruction");
    events.clear();

    // Window-recreating variant (WGL/GLX/SDL policy).
    root.RetireSessionForReconstruction(true);
    Expect(events.back() == "platform-stop",
           "window policy releases platform resources");
    host.CreateMainWindow();
    Expect(root.PrepareReconstructedSession(true) == 0,
           "backend recreates on the fresh window");
    RecordingModule s4("s4", events);
    root.RegisterSessionModule(s4);
    Expect(root.Session().Activate() == 0, "second reconstruction activates");
    root.SetFramePlan({&p1, &s4, &p2});
    root.Finalize();
}

void TestRootReconstructionFailure() {
    {
        std::vector<std::string> events;
        StubHost host(events);
        RecordingModule p1("p1", events), s1("s1", events);
        RuntimeRoot root;
        root.SetHost(host, host, host);
        root.RegisterPersistentModule(p1);
        root.RegisterSessionModule(s1);
        root.SetFramePlan({&p1, &s1});
        Expect(root.Initialize() == 0, "startup before failed reconstruction");
        const auto generation = root.Session().Generation();
        events.clear();

        root.RetireSessionForReconstruction(false);
        host.graphicsResult = 13;
        Expect(root.PrepareReconstructedSession(false) == 13,
               "reconstruction graphics failure propagates");
        Expect(!root.Session().IsActive(),
               "failed reconstruction leaves no session");
        Expect(root.Session().Generation() == generation + 1,
               "retirement bumped the generation; failure publishes nothing");
        bool persistentUntouched = true;
        for (const auto& event : events) {
            if (event == "init p1" || event == "stop p1") {
                persistentUntouched = false;
            }
        }
        Expect(persistentUntouched,
               "failed reconstruction never touches persistent services");
        events.clear();
        root.Finalize();
        Expect(events == std::vector<std::string>{"drain", "stop p1", "gui-stop",
                                                  "platform-stop"},
               "Finalize after failed reconstruction finalizes persistent "
               "services, the kept GUI binding, and platform");
    }
    {
        std::vector<std::string> events;
        StubHost host(events);
        RecordingModule p1("p1", events), s1("s1", events);
        RuntimeRoot root;
        root.SetHost(host, host, host);
        root.RegisterPersistentModule(p1);
        root.RegisterSessionModule(s1);
        Expect(root.Initialize() == 0, "startup before GUI failure");
        events.clear();

        root.RetireSessionForReconstruction(true);
        host.CreateMainWindow();
        host.guiResult = 19;
        Expect(root.PrepareReconstructedSession(true) == 19,
               "reconstruction GUI platform failure propagates");
        Expect(events == std::vector<std::string>{
                     "drain", "stop s1", "gui-stop", "graphics-stop",
                     "platform-stop", "window-create",
                     "graphics-init", "gui-init", "graphics-stop"},
               "GUI failure rolls back the recreated graphics state");
        events.clear();

        // Recovery: a second attempt succeeds without restarting the runtime.
        host.guiResult = 0;
        Expect(root.PrepareReconstructedSession(true) == 0,
               "reconstruction retry succeeds");
        RecordingModule s2("s2", events);
        root.RegisterSessionModule(s2);
        Expect(root.Session().Activate() == 0, "recovered session activates");
        root.SetFramePlan({&p1, &s2});
        events.clear();
        root.Tick();
        Expect(events == std::vector<std::string>{"events", "tick p1", "tick s2"},
               "persistent services resume with the recovered session");
        root.Finalize();
    }
}

void TestRootMemorySystemOwnership() {
    std::vector<std::string> events;
    RecordingModule persistent("persistent", events);
    RecordingModule session("session", events);
    MemorySystem::Config config;
    config.reportHandler = [](const std::string&) {};

    {
        RuntimeRoot root;
        Expect(root.GetMemorySystem() == nullptr,
               "no allocator stack before initialization");
        Expect(root.InitializeMemorySystem(config),
               "engine allocator stack configures");
        MemorySystem* system = root.GetMemorySystem();
        Expect(system != nullptr, "root owns the allocator stack");
        Expect(root.InitializeMemorySystem(config),
               "repeated initialization keeps the existing system");
        Expect(root.GetMemorySystem() == system,
               "allocator stack is not recreated");

        void* block = EngineAllocate(64);
        Expect(block != nullptr, "hook serves allocations");
        Expect(EngineAllocationCount() == 1, "block is routed to the engine");
        EngineFree(block);
        Expect(EngineAllocationCount() == 0, "block returns to the engine");

        root.PersistentModules().Register(persistent);
        Expect(root.PersistentModules().Initialize() == 0, "persistent up");
        for (int cycle = 0; cycle < 2; ++cycle) {
            root.Session().Register(session);
            Expect(root.Session().Activate() == 0, "session up");
            root.BeginFrame();
            root.Session().Retire();
        }
        Expect(root.GetMemorySystem() == system,
               "session replacement never releases the allocator stack");
        block = EngineAllocate(64);
        Expect(EngineAllocationCount() == 1,
               "hook still routes after session replacement");
        EngineFree(block);
        root.PersistentModules().Finalize();
    }
    Expect(EngineAllocationCount() == 0,
           "root destruction clears the hook without leaking engine blocks");
    void* heapBlock = EngineAllocate(64);
    Expect(heapBlock != nullptr && EngineAllocationCount() == 0,
           "allocations fall back to the default heap after release");
    EngineFree(heapBlock);

    {
        RuntimeRoot root;
        MemorySystem::Config unconfigured;
        unconfigured.largeCapacity = 0;
        unconfigured.frameCapacity = 0;
        unconfigured.reportHandler = [](const std::string&) {};
        Expect(!root.InitializeMemorySystem(unconfigured),
               "unconfigured allocator stack is non-fatal");
        Expect(root.GetMemorySystem() == nullptr,
               "no system retained when unconfigured");
        root.BeginFrame();
    }
}
}  // namespace

int main() {
    TestPersistentUnaffectedBySessionCycles();
    TestSessionFailureRollbackAndRetry();
    TestGenerationSemantics();
    TestFrameOrderPreserved();
    TestRootStartupAndTeardownOrder();
    TestRootFailurePaths();
    TestRootReconstructionCycle();
    TestRootReconstructionFailure();
    TestRootMemorySystemOwnership();
    return failures == 0 ? 0 : 1;
}
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
    TestRootMemorySystemOwnership();
    return failures == 0 ? 0 : 1;
}
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "RuntimeModuleGroup.hpp"

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

void TestSuccessfulLifecycle() {
    std::vector<std::string> events;
    RecordingModule first("first", events), second("second", events);
    RuntimeModuleGroup modules;
    modules.Register(first);
    modules.Register(second);
    modules.Tick();
    Expect(events.empty(), "no updates before initialization");
    Expect(modules.Initialize() == 0, "initialization succeeds");
    modules.Tick();
    modules.Finalize();
    Expect(events == std::vector<std::string>{
                         "init first", "init second", "tick first",
                         "tick second", "stop second", "stop first"},
           "forward initialization/update and reverse finalization");
    events.clear();
    modules.Finalize();
    modules.Tick();
    Expect(events.empty(), "finalization is not repeated");
}

void TestFailureRollbackAndRetry() {
    for (int failingIndex = 0; failingIndex < 3; ++failingIndex) {
        std::vector<std::string> events;
        RecordingModule first("first", events), second("second", events),
            third("third", events);
        RecordingModule* ordered[] = {&first, &second, &third};
        RuntimeModuleGroup modules;
        for (auto* module : ordered) modules.Register(*module);
        ordered[failingIndex]->result = 17;
        Expect(modules.Initialize() == 17, "preserve initialization error");
        std::vector<std::string> expected;
        const std::string names[] = {"first", "second", "third"};
        for (int i = 0; i <= failingIndex; ++i)
            expected.push_back("init " + names[i]);
        for (int i = failingIndex - 1; i >= 0; --i)
            expected.push_back("stop " + names[i]);
        Expect(events == expected,
               "rollback only initialized modules, in reverse order");
        events.clear();
        modules.Tick();
        Expect(events.empty(), "failed group never updates");
        ordered[failingIndex]->result = 0;
        Expect(modules.Initialize() == 0, "retry after rollback succeeds");
        modules.Finalize();
        Expect(events == std::vector<std::string>{
                             "init first", "init second", "init third",
                             "stop third", "stop second", "stop first"},
               "retry retains registrations and finalizes exactly once");
        events.clear();
        modules.Finalize();
        Expect(events.empty(), "rollback/retry does not double finalize");
    }
}

void TestRegistrationGuards() {
    std::vector<std::string> events;
    RecordingModule first("first", events), second("second", events);
    RuntimeModuleGroup modules;
    modules.Register(first);
    bool rejected = false;
    try {
        modules.Register(first);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Expect(rejected, "reject duplicate registrations");
    Expect(modules.Initialize() == 0, "initialize guarded group");
    rejected = false;
    try {
        modules.Register(second);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    Expect(rejected, "reject registration after initialization");
    Expect(modules.Initialize() != 0, "reject repeated initialization");
    modules.Finalize();
    Expect(events == std::vector<std::string>{"init first", "stop first"},
           "rejected operations do not modify lifecycle");
}
}  // namespace

int main() {
    TestSuccessfulLifecycle();
    TestFailureRollbackAndRetry();
    TestRegistrationGuards();
    return failures == 0 ? 0 : 1;
}

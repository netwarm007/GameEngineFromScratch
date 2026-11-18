#include <iostream>
#include <string>
#include <vector>

#include "GameLogic.hpp"
#include "InputManager.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class RecordingLogic : public GameLogic {
   public:
    std::vector<std::string> events;
    struct Drag {
        int id;
        float x;
        float y;
    };
    std::vector<Drag> drags;

    void OnUpKeyDown() override { events.push_back("up down"); }
    void OnUpKey() override { events.push_back("up first"); }
    void OnUpKeyUp() override { events.push_back("up release"); }
    void OnDownKeyDown() override { events.push_back("down down"); }
    void OnDownKey() override { events.push_back("down first"); }
    void OnDownKeyUp() override { events.push_back("down release"); }
    void OnLeftKeyDown() override { events.push_back("left down"); }
    void OnLeftKey() override { events.push_back("left first"); }
    void OnLeftKeyUp() override { events.push_back("left release"); }
    void OnRightKeyDown() override { events.push_back("right down"); }
    void OnRightKey() override { events.push_back("right first"); }
    void OnRightKeyUp() override { events.push_back("right release"); }
    void OnButton1Down() override { events.push_back("button down"); }
    void OnButton1Up() override { events.push_back("button release"); }
    void OnAnalogStick(int id, float x, float y) override {
        drags.push_back({id, x, y});
    }
};

void TestArrowSemantics() {
    RecordingLogic logic;
    InputManager input(logic);
    Expect(input.Initialize() == 0, "input initializes without an application");
    struct Arrow {
        const char* name;
        void (InputManager::*down)();
        void (InputManager::*up)();
    };
    const Arrow arrows[] = {
        {"up", &InputManager::UpArrowKeyDown, &InputManager::UpArrowKeyUp},
        {"down", &InputManager::DownArrowKeyDown, &InputManager::DownArrowKeyUp},
        {"left", &InputManager::LeftArrowKeyDown, &InputManager::LeftArrowKeyUp},
        {"right", &InputManager::RightArrowKeyDown, &InputManager::RightArrowKeyUp}};
    std::vector<std::string> expected;
    for (const auto& arrow : arrows) {
        (input.*arrow.down)();
        expected.push_back(std::string(arrow.name) + " down");
        expected.push_back(std::string(arrow.name) + " first");
    }
    for (const auto& arrow : arrows) {
        (input.*arrow.down)();
        expected.push_back(std::string(arrow.name) + " down");
    }
    Expect(logic.events == expected,
           "simultaneous arrows repeat down but emit first-press independently");
    for (const auto& arrow : arrows) {
        (input.*arrow.up)();
        (input.*arrow.down)();
        expected.push_back(std::string(arrow.name) + " release");
        expected.push_back(std::string(arrow.name) + " down");
        expected.push_back(std::string(arrow.name) + " first");
    }
    Expect(logic.events == expected, "release rearms each arrow in callback order");

    for (int lifecycle = 0; lifecycle < 2; ++lifecycle) {
        input.Finalize();
        input.Finalize();
        Expect(input.Initialize() == 0, "input supports a fresh lifecycle");
        logic.events.clear();
        for (const auto& arrow : arrows) (input.*arrow.down)();
        Expect(logic.events == std::vector<std::string>{
                   "up down", "up first", "down down", "down first",
                   "left down", "left first", "right down", "right first"},
               "fresh lifecycle clears all held-key state");
    }
    input.Finalize();
}

void TestButtonsAndDrags() {
    RecordingLogic logic;
    InputManager input(logic);
    input.Initialize();
    input.AsciiKeyDown('u');
    input.AsciiKeyDown('u');
    input.AsciiKeyUp('u');
    input.AsciiKeyDown('d');
    input.AsciiKeyUp('d');
    input.AsciiKeyDown('r');
    input.AsciiKeyUp('r');
    input.LeftMouseButtonDown();
    input.LeftMouseButtonUp();
    input.RightMouseButtonDown();
    input.RightMouseButtonUp();
    input.Tick();
    Expect(logic.events == std::vector<std::string>{
               "button down", "button down", "button release"},
           "button mapping, repeats and reserved/no-op events are preserved");
    input.LeftMouseDrag(-12, 34);
    input.RightMouseDrag(56, -78);
    Expect(logic.drags.size() == 2 && logic.drags[0].id == 0 &&
               logic.drags[0].x == -12.0f && logic.drags[0].y == 34.0f &&
               logic.drags[1].id == 1 && logic.drags[1].x == 56.0f &&
               logic.drags[1].y == -78.0f,
           "mouse drag preserves analog id, signed deltas and ordering");
    input.Finalize();
}
}  // namespace

int main() {
    TestArrowSemantics();
    TestButtonsAndDrags();
    return failures ? 1 : 0;
}

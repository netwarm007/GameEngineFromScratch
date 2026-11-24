#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "Framework/Manager/RhiStartup.hpp"
#include "RhiSwitch.hpp"

namespace My {
class RuntimeRoot;
class BaseApplication;
class InputManager;
class IGraphicsManager;
class IPipelineStateManager;
class ISceneManager;
class IAssetReader;
class MemorySystem;
struct RhiStartupComposition;

// One scheduled backend switch at a frame boundary (shared by Viewer/Billiard).
struct RhiSwitchStep {
    RhiBackend backend;
    int frame;
};

// Per-frame runtime-RHI switch orchestration shared by the application entry
// points (M5.3 composition reuse): it advances the root, fires scheduled
// switches, processes the frame boundary, reconciles the frame plan and session
// managers after a switch or reconstruction, and tracks simulation pause. The
// app supplies hooks for its frame-plan rebuild and any post-rebind actions
// (Viewer re-binds the selection provider and updates the window title).
class RhiFrameDriver {
public:
    struct Hooks {
        // Rebuild and apply the frame plan for the given pause state.
        std::function<void(bool paused)> applyFramePlan;
        // Called after the session managers are rebound to the fresh session
        // (after a switch or reconstruction). May be empty.
        std::function<void()> onSessionRebound;
    };

    RhiFrameDriver(RhiSwitchController& controller, RuntimeRoot& root,
                   RhiStartupComposition& composition, BaseApplication& app,
                   InputManager& input, ISceneManager& scene,
                   IAssetReader& assets, IGraphicsManager*& graphicsManager,
                   IPipelineStateManager*& pipelineStateManager,
                   std::function<MemorySystem*()> memoryProvider,
                   std::vector<RhiSwitchStep> schedule, int reconstructCycles,
                   Hooks hooks);

    // Advance one frame. Returns 1 when an exit was requested (failed switch /
    // reconstruction), else 0.
    int Tick();

private:
    void RebindSession();

    RhiSwitchController& m_Controller;
    RuntimeRoot& m_Root;
    RhiStartupComposition& m_Composition;
    BaseApplication& m_App;
    InputManager& m_Input;
    ISceneManager& m_Scene;
    IAssetReader& m_Assets;
    IGraphicsManager*& m_GraphicsManager;
    IPipelineStateManager*& m_PipelineStateManager;
    std::function<MemorySystem*()> m_MemoryProvider;
    std::vector<RhiSwitchStep> m_Schedule;
    int m_ReconstructCycles;
    Hooks m_Hooks;
    std::size_t m_NextSwitch = 0;
    int m_FrameCount = 0;
    bool m_SimulationPaused = false;
};
}  // namespace My
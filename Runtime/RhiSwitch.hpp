#pragma once

#include <functional>
#include <string>

#include "Framework/Manager/RuntimeRoot.hpp"
#include "RhiComposition.hpp"

namespace My {
class GfxConfiguration;
class ISceneManager;
class IAssetReader;

enum class RhiSwitchState { Running, FirstFrame, ExitRequested };

enum class RhiSwitchRequestResult {
    Accepted,
    AlreadyActive,
    Overlapping,
    NotCompiled,
    Unavailable
};

std::string_view RhiSwitchRequestResultName(RhiSwitchRequestResult result);

// Frame-boundary cross-backend switch controller (M4.2). One request at a
// time; selecting the active backend is a no-op; preflight keeps the old
// session on failure; failure after retirement attempts one previous-
// backend reconstruction and reports the failed switch even when recovery
// succeeds; recovery failure requests an orderly exit with both errors.
// The retired session is never reused. Simulation stays paused until the
// restored scene completes a rendered/presented frame (FirstFrame gate);
// a zero-size framebuffer neither publishes success nor busy-waits.
class RhiSwitchController {
   public:
    using BackendFactory =
        std::function<RhiBackendResult(RhiBackend, BaseApplication&)>;
    using SessionFactory = std::function<RhiSessionResult(
        RhiBackend, IRenderBackend&, ISceneManager&, IAssetReader&,
        std::function<MemorySystem*()>)>;
    using AvailabilityProbe = std::function<std::vector<RhiBackendAvailability>()>;
    using FramebufferQuery = std::function<void(uint32_t&, uint32_t&)>;

    RhiSwitchController(RuntimeRoot& root, RhiStartupComposition& composition,
                        GfxConfiguration& configuration,
                        ISceneManager& sceneManager, IAssetReader& assetReader,
                        std::function<MemorySystem*()> memoryProvider,
                        BackendFactory backendFactory = {},
                        SessionFactory sessionFactory = {},
                        AvailabilityProbe availabilityProbe = {},
                        FramebufferQuery framebufferQuery = {});

    // Preflight and queue a switch for the next complete frame boundary.
    RhiSwitchRequestResult Request(RhiBackend target);

    // Drive the state machine after root.Tick() (complete frame boundary,
    // after main and secondary viewport submission).
    void ProcessFrameBoundary();

    RhiSwitchState State() const { return m_State; }
    bool IsSimulationPaused() const { return m_State != RhiSwitchState::Running; }
    bool ExitRequested() const { return m_State == RhiSwitchState::ExitRequested; }
    RhiBackend ActiveBackend() const { return m_Composition.backend; }
    const std::string& LastError() const { return m_LastError; }

    // The session modules changed; the caller must rebuild its frame plan.
    bool ConsumeFramePlanDirty();

   private:
    // Returns 0 on success; on failure the session is absent and the error
    // is recorded. Never recovers internally.
    int ActivateBackend(RhiBackend target);
    void RecoverPrevious(RhiBackend previous, const std::string& switchError);

    RuntimeRoot& m_Root;
    RhiStartupComposition& m_Composition;
    GfxConfiguration& m_Configuration;
    ISceneManager& m_SceneManager;
    IAssetReader& m_AssetReader;
    std::function<MemorySystem*()> m_MemoryProvider;
    BackendFactory m_BackendFactory;
    SessionFactory m_SessionFactory;
    AvailabilityProbe m_AvailabilityProbe;
    FramebufferQuery m_FramebufferQuery;

    RhiSwitchState m_State = RhiSwitchState::Running;
    bool m_Pending = false;
    RhiBackend m_PendingTarget = RhiBackend::OpenGL;
    bool m_FramePlanDirty = false;
    std::string m_LastError;
};

}  // namespace My
#include "RhiSwitch.hpp"

#include <cstdlib>
#include <iostream>
#include <utility>

#include "Framework/Common/GfxConfiguration.hpp"
#include "Framework/Interface/IAssetReader.hpp"
#include "Framework/Interface/ISceneManager.hpp"

using namespace My;

std::string_view My::RhiSwitchRequestResultName(RhiSwitchRequestResult result) {
    switch (result) {
        case RhiSwitchRequestResult::Accepted:
            return "accepted";
        case RhiSwitchRequestResult::AlreadyActive:
            return "already-active";
        case RhiSwitchRequestResult::Overlapping:
            return "overlapping";
        case RhiSwitchRequestResult::NotCompiled:
            return "not-compiled";
        case RhiSwitchRequestResult::Unavailable:
            return "unavailable";
    }
    return "unknown";
}

RhiSwitchController::RhiSwitchController(
    RuntimeRoot& root, RhiStartupComposition& composition,
    GfxConfiguration& configuration, ISceneManager& sceneManager,
    IAssetReader& assetReader, std::function<MemorySystem*()> memoryProvider,
    BackendFactory backendFactory, SessionFactory sessionFactory,
    AvailabilityProbe availabilityProbe, FramebufferQuery framebufferQuery)
    : m_Root(root),
      m_Composition(composition),
      m_Configuration(configuration),
      m_SceneManager(sceneManager),
      m_AssetReader(assetReader),
      m_MemoryProvider(std::move(memoryProvider)),
      m_BackendFactory(backendFactory ? std::move(backendFactory)
                                      : BackendFactory(CreateRhiBackendObject)),
      m_SessionFactory(sessionFactory
                           ? std::move(sessionFactory)
                           : SessionFactory(CreateRhiSessionModules)),
      m_AvailabilityProbe(availabilityProbe
                              ? std::move(availabilityProbe)
                              : AvailabilityProbe(ProbeRhiStartupAvailability)),
      m_FramebufferQuery(std::move(framebufferQuery)) {}

RhiSwitchRequestResult RhiSwitchController::Request(RhiBackend target) {
    if (m_State != RhiSwitchState::Running || m_Pending) {
        return RhiSwitchRequestResult::Overlapping;
    }
    if (target == m_Composition.backend) {
        return RhiSwitchRequestResult::AlreadyActive;
    }
    for (const auto& availability : m_AvailabilityProbe()) {
        if (availability.backend != target) continue;
        if (availability.state == RhiAvailability::NotCompiled) {
            m_LastError = availability.reason;
            return RhiSwitchRequestResult::NotCompiled;
        }
        if (availability.state == RhiAvailability::RuntimeUnavailable) {
            m_LastError = availability.reason;
            return RhiSwitchRequestResult::Unavailable;
        }
    }
    m_Pending = true;
    m_PendingTarget = target;
    return RhiSwitchRequestResult::Accepted;
}

void RhiSwitchController::ProcessFrameBoundary() {
    if (m_State == RhiSwitchState::ExitRequested) return;

    if (m_State == RhiSwitchState::FirstFrame) {
        // Publish success only after a rendered/presented frame at a usable
        // framebuffer size; a minimized/zero-size window keeps waiting
        // without busy-looping (one check per frame boundary).
        uint32_t width = 0, height = 0;
        if (m_FramebufferQuery) {
            m_FramebufferQuery(width, height);
        } else {
            m_Root.Application()->GetFramebufferSize(width, height);
        }
        if (width > 0 && height > 0) {
            m_State = RhiSwitchState::Running;
            m_LastError.clear();
            std::cout << "RHI switch complete: "
                      << RhiBackendName(m_Composition.backend) << '\n';
        }
        return;
    }

    if (!m_Pending) return;
    m_Pending = false;

    const RhiBackend previous = m_Composition.backend;
    const RhiBackend target = m_PendingTarget;
    if (ActivateBackend(target) != 0) {
        RecoverPrevious(previous, m_LastError);
        return;
    }
    m_State = RhiSwitchState::FirstFrame;
}

int RhiSwitchController::ActivateBackend(RhiBackend target) {
    // If a prior failed switch already tore the window down, recreate it even
    // for a backend that normally keeps its window (e.g. recovering to D3D12
    // after an OpenGL switch failed post-retirement): initializing on a missing
    // window would otherwise fail and defeat recovery.
    const bool windowAbsent = !m_Composition.application->GetMainWindowHandler();
    const bool recreateWindow =
        m_Composition.recreateWindowOnReconstruction ||
        RhiBackendRequiresWindowRecreation(target) ||
        windowAbsent;

    // Clear the host's borrowed session pointers before retirement: window
    // teardown routes resize messages through them.
    m_Composition.application->BindSessionManagers(nullptr, nullptr);
    m_Root.RetireSessionForReconstruction(recreateWindow);
    m_Composition.graphicsManager.reset();
    m_Composition.pipelineStateManager.reset();

    // Verification-only failure injection (GEFS_RHI_SWITCH_FAIL_ONCE=1): force
    // a single post-retirement failure to exercise recovery and the error UI.
    // Self-clearing so subsequent switches succeed; never affects normal runs.
    if (const char* failOnce = std::getenv("GEFS_RHI_SWITCH_FAIL_ONCE");
        failOnce && failOnce[0] == '1') {
#ifdef _WIN32
        _putenv_s("GEFS_RHI_SWITCH_FAIL_ONCE", "");
#else
        unsetenv("GEFS_RHI_SWITCH_FAIL_ONCE");
#endif
        m_LastError = "GEFS_RHI_SWITCH_FAIL_ONCE: injected ActivateBackend failure";
        std::cerr << "RHI switch: " << m_LastError << '\n';
        return -1;
    }

    auto backendResult = m_BackendFactory(target, *m_Composition.application);
    if (!backendResult.backend) {
        // The retired backend object stays alive (finalized) so recovery can
        // still drain through the root's backend pointer; it is replaced
        // only once the target backend exists.
        m_LastError = "backend creation failed: " + backendResult.error;
        std::cerr << "RHI switch: " << m_LastError << '\n';
        return -1;
    }

    m_Root.Platform()->OnBackendSwitch(target);
    if (recreateWindow) backendResult.backend->CreateMainWindow();

    m_Composition.sessionBackend = std::move(backendResult.backend);
    m_Composition.backend = target;
    m_Composition.recreateWindowOnReconstruction =
        backendResult.recreateWindowOnReconstruction;
    m_Composition.fixOpenGLPerspectiveMatrix =
        backendResult.fixOpenGLPerspectiveMatrix;
    m_Configuration.fixOpenGLPerspectiveMatrix =
        backendResult.fixOpenGLPerspectiveMatrix;
    m_Composition.application->SetRenderConvention(
        backendResult.fixOpenGLPerspectiveMatrix);

    m_Root.SetHost(*m_Composition.application, *m_Root.Platform(),
                   *m_Composition.sessionBackend);
    const int prepared = m_Root.PrepareReconstructedSession(recreateWindow);
    if (prepared != 0) {
        m_LastError = "native initialization failed (error " +
                      std::to_string(prepared) + ")";
        std::cerr << "RHI switch: " << m_LastError << '\n';
        return -1;
    }

    auto session = m_SessionFactory(target, *m_Composition.sessionBackend,
                                    m_SceneManager, m_AssetReader,
                                    m_MemoryProvider);
    if (!session.modules.graphicsManager ||
        !session.modules.pipelineStateManager) {
        m_LastError = "session creation failed: " + session.error;
        std::cerr << "RHI switch: " << m_LastError << '\n';
        return -1;
    }
    m_Composition.pipelineStateManager =
        std::move(session.modules.pipelineStateManager);
    m_Composition.graphicsManager = std::move(session.modules.graphicsManager);
    m_Composition.application->BindSessionManagers(
        m_Composition.graphicsManager.get(),
        m_Composition.pipelineStateManager.get());
    m_Root.RegisterSessionModule(*m_Composition.graphicsManager);
    m_Root.RegisterSessionModule(*m_Composition.pipelineStateManager);
    const int activated = m_Root.Session().Activate(
        [this] { m_Root.Backend()->WaitForGraphicsIdle(); });
    if (activated != 0) {
        m_Composition.graphicsManager.reset();
        m_Composition.pipelineStateManager.reset();
        m_LastError =
            "session activation failed (error " + std::to_string(activated) + ")";
        std::cerr << "RHI switch: " << m_LastError << '\n';
        return -1;
    }
    m_Root.Backend()->OnSessionActivated();
    m_FramePlanDirty = true;
    return 0;
}

void RhiSwitchController::RecoverPrevious(RhiBackend previous,
                                          const std::string& switchError) {
    std::cerr << "RHI switch to " << RhiBackendName(m_PendingTarget)
              << " failed: " << switchError << "; recovering "
              << RhiBackendName(previous) << '\n';
    if (ActivateBackend(previous) == 0) {
        // The failed switch is reported even when recovery succeeds.
        m_LastError = "switch failed: " + switchError + "; recovered " +
                      std::string(RhiBackendName(previous));
        m_State = RhiSwitchState::FirstFrame;
        return;
    }
    m_LastError = "switch failed: " + switchError + "; recovery failed: " +
                  m_LastError;
    std::cerr << "RHI switch: " << m_LastError << "; exiting orderly\n";
    m_State = RhiSwitchState::ExitRequested;
}

bool RhiSwitchController::ConsumeFramePlanDirty() {
    const bool dirty = m_FramePlanDirty;
    m_FramePlanDirty = false;
    return dirty;
}
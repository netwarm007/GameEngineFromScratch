#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Framework/Interface/IGraphicsManager.hpp"
#include "Framework/Interface/IRenderBackend.hpp"
#include "Framework/Interface/IPipelineStateManager.hpp"
#include "Framework/Manager/BaseApplication.hpp"
#include "Framework/Manager/RhiStartup.hpp"

namespace My {
class ISceneManager;
class IAssetReader;
class MemorySystem;
class RuntimeRoot;

// Owned startup composition for one selected backend. Members are declared in
// dependency order so destruction retires graphics, then pipeline state, then
// the application that owns the backend session. Construction never creates a
// window or initializes graphics; Initialize() remains the caller's step.
// The application object is the persistent host: it survives same-backend
// session reconstruction, which replaces only the session-owned managers.
struct RhiStartupComposition {
    RhiBackend backend = RhiBackend::OpenGL;
    bool fixOpenGLPerspectiveMatrix = false;
    // WGL pixel format and GLX/SDL contexts are window-lifetime bound, so
    // OpenGL reconstruction recreates the native window; other backends
    // rebuild their drawable on the existing one.
    bool recreateWindowOnReconstruction = false;
    std::unique_ptr<BaseApplication> application;
    // Session-owned backend lifecycle (M4.1 detachment). Null only for the
    // legacy GLX fallback where the application implements the backend
    // itself; the host application object persists across reconstruction.
    std::unique_ptr<IRenderBackend> sessionBackend;
    std::unique_ptr<IPipelineStateManager> pipelineStateManager;
    std::unique_ptr<IGraphicsManager> graphicsManager;
};

struct RhiCompositionResult {
    std::unique_ptr<RhiStartupComposition> composition;
    std::string error;
};

// Replaceable session-owned modules for one backend on the persistent host.
// Construction is window/GPU-free; the memory provider resolves the root's
// allocator stack lazily (it is created after composition construction).
struct RhiSessionModules {
    std::unique_ptr<IPipelineStateManager> pipelineStateManager;
    std::unique_ptr<IGraphicsManager> graphicsManager;
};

struct RhiSessionResult {
    RhiSessionModules modules;
    std::string error;
};

RhiSessionResult CreateRhiSessionModules(
    RhiBackend backend, IRenderBackend& backendObject,
    ISceneManager& sceneManager, IAssetReader& assetReader,
    std::function<MemorySystem*()> memoryProvider = {});

// Same-backend reconstruction at a complete frame boundary: retire the
// session modules and native backend state (recreating the window when the
// backend requires it), recreate them, and replace the composition's
// session modules. Persistent services and the host application object are
// never touched. The caller must rebuild its frame plan afterwards and must
// not tick while reconstruction runs. Returns 0 on success; on failure the
// session is absent and the caller reports the error (recovery policy is
// M4 scope).
int ReconstructRhiSession(RuntimeRoot& root,
                          RhiStartupComposition& composition,
                          ISceneManager& sceneManager, IAssetReader& assetReader,
                          std::function<MemorySystem*()> memoryProvider = {});

// Typed, explicit dispatch to the factories compiled into this application.
// Referencing every included factory here retains its implementation in the
// executable; a backend that is not compiled fails explicitly instead of
// selecting another one. The optional memory provider resolves the persistent
// allocator stack (the runtime root); when absent, the D3D12 session resolves
// the host application's own stack (legacy startup path).
RhiCompositionResult CreateRhiStartupComposition(
    RhiBackend backend, GfxConfiguration& configuration,
    ISceneManager& sceneManager, IAssetReader& assetReader,
    std::function<MemorySystem*()> memoryProvider = {});

// Compiled inclusion plus runtime prerequisite evidence. ReadyForInitialization
// is not successful device/window initialization; no backend is probed by
// creating a device here.
std::vector<RhiBackendAvailability> ProbeRhiStartupAvailability();

// Platform and windowing host labels for the active composition, used by
// applications to show the current platform/backend combination in the title.
std::string_view RhiStartupPlatformName();
std::string_view RhiStartupHostName(RhiBackend backend);

}  // namespace My
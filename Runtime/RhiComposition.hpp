#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Framework/Interface/IGraphicsManager.hpp"
#include "Framework/Interface/IPipelineStateManager.hpp"
#include "Framework/Manager/BaseApplication.hpp"
#include "Framework/Manager/RhiStartup.hpp"

namespace My {
class ISceneManager;
class IAssetReader;

// Owned startup composition for one selected backend. Members are declared in
// dependency order so destruction retires graphics, then pipeline state, then
// the application that owns the backend session. Construction never creates a
// window or initializes graphics; Initialize() remains the caller's step.
struct RhiStartupComposition {
    RhiBackend backend = RhiBackend::OpenGL;
    bool fixOpenGLPerspectiveMatrix = false;
    std::unique_ptr<BaseApplication> application;
    std::unique_ptr<IPipelineStateManager> pipelineStateManager;
    std::unique_ptr<IGraphicsManager> graphicsManager;
};

struct RhiCompositionResult {
    std::unique_ptr<RhiStartupComposition> composition;
    std::string error;
};

// Typed, explicit dispatch to the factories compiled into this application.
// Referencing every included factory here retains its implementation in the
// executable; a backend that is not compiled fails explicitly instead of
// selecting another one.
RhiCompositionResult CreateRhiStartupComposition(
    RhiBackend backend, GfxConfiguration& configuration,
    ISceneManager& sceneManager, IAssetReader& assetReader);

// Compiled inclusion plus runtime prerequisite evidence. ReadyForInitialization
// is not successful device/window initialization; no backend is probed by
// creating a device here.
std::vector<RhiBackendAvailability> ProbeRhiStartupAvailability();

// Platform and windowing host labels for the active composition, used by
// applications to show the current platform/backend combination in the title.
std::string_view RhiStartupPlatformName();
std::string_view RhiStartupHostName(RhiBackend backend);

}  // namespace My
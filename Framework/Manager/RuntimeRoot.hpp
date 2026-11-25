#pragma once

#include <memory>
#include <vector>

#include "IApplication.hpp"
#include "IPlatformHost.hpp"
#include "IRenderBackend.hpp"
#include "MemorySystem.hpp"
#include "RenderSession.hpp"
#include "RuntimeModuleGroup.hpp"

namespace My {

// Persistent composition root: long-lived CPU services and the engine
// memory system outlive every render session. The root owns one replaceable
// session slot and orchestrates the persistent host (window/event routing,
// GUI platform binding, backend-native lifecycle) around it. The host
// application object persists across session replacement, so every module's
// borrowed application pointer remains valid.
//
// Frame participant order stays caller-composed (SetFramePlan + TickFrame),
// so introducing the session boundary does not reopen the frame scheduler.
class RuntimeRoot {
   public:
    RuntimeRoot();
    // The caller must Finalize before destruction; borrowed modules and the
    // host must outlive the root.
    ~RuntimeRoot();
    RuntimeRoot(const RuntimeRoot&) = delete;
    RuntimeRoot& operator=(const RuntimeRoot&) = delete;

    // The persistent host application implements all three interfaces.
    void SetHost(IApplication& application, IPlatformHost& platform,
                 IRenderBackend& backend);
    IApplication* Application() { return m_Application; }
    IPlatformHost* Platform() { return m_Platform; }
    IRenderBackend* Backend() { return m_Backend; }

    // Registration sets the borrowed application pointer. Persistent
    // modules are long-lived CPU services; session modules are the
    // replaceable render side (graphics/pipeline managers).
    void RegisterPersistentModule(IRuntimeModule& module);
    void RegisterSessionModule(IRuntimeModule& module);

    // Borrowed frame participants in the desired tick order; the caller
    // rebuilds the plan when session modules are replaced.
    void SetFramePlan(std::vector<IRuntimeModule*> plan);

    // Startup: backend graphics resources -> GUI platform binding -> engine
    // memory system -> persistent modules -> session activation. On failure
    // the caller runs Finalize, which releases everything idempotently.
    int Initialize();
    // Teardown: GPU drain -> session retirement -> persistent finalization
    // -> GUI platform unbind -> backend graphics -> platform resources ->
    // memory system.
    void Finalize();

    // One frame: platform events, frame-arena recycle, module updates.
    void Tick();

    // Same-backend reconstruction primitives. The composition layer drives
    // the cycle (module factories and window policy live there): retire the
    // session and native backend state, recreate, then register and activate
    // fresh session modules. Persistent services are never touched.
    void RetireSessionForReconstruction(bool recreateWindow);
    int PrepareReconstructedSession(bool windowRecreated);

    RuntimeModuleGroup& PersistentModules() { return m_Persistent; }
    RenderSessionSlot& Session() { return m_Session; }

    // Optional engine allocator stack, owned here so session replacement
    // cannot release allocators that persistent services still use. The
    // root assumes exclusive ownership of the process-wide allocation hook.
    // Non-fatal: an unconfigured system falls back to the default heap.
    bool InitializeMemorySystem(const MemorySystem::Config& config);
    MemorySystem* GetMemorySystem() { return m_pMemorySystem.get(); }

    void BeginFrame() {
        if (m_pMemorySystem) m_pMemorySystem->BeginFrame();
        if (m_Backend) m_Backend->BeginFrame();
    }

   private:
    void ReleaseMemorySystem();
    IApplication* m_Application = nullptr;
    IPlatformHost* m_Platform = nullptr;
    IRenderBackend* m_Backend = nullptr;
    bool m_GraphicsInitialized = false;
    bool m_GuiPlatformInitialized = false;
    RuntimeModuleGroup m_Persistent;
    RenderSessionSlot m_Session;
    std::vector<IRuntimeModule*> m_FramePlan;
    std::unique_ptr<MemorySystem> m_pMemorySystem;
};

}  // namespace My
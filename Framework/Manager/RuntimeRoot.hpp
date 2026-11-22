#pragma once

#include <memory>

#include "MemorySystem.hpp"
#include "RenderSession.hpp"
#include "RuntimeModuleGroup.hpp"

namespace My {

// Persistent composition root: long-lived CPU services and the engine
// memory system outlive every render session. The root owns one replaceable
// session slot; reconstruction never re-initializes persistent modules or
// tears down allocators they still use. Frame participant order stays the
// caller's choice (TickFrame over a caller-composed span), so introducing
// the session boundary does not reopen the frame scheduler.
//
// Native wiring lands with M3.4; until then applications keep their existing
// startup paths and this root is exercised headlessly.
class RuntimeRoot {
   public:
    RuntimeRoot();
    // The caller must retire the session and finalize persistent modules
    // before destruction; borrowed modules must outlive the root.
    ~RuntimeRoot();
    RuntimeRoot(const RuntimeRoot&) = delete;
    RuntimeRoot& operator=(const RuntimeRoot&) = delete;

    // Long-lived CPU services (scene, assets, physics, animation, game
    // logic, input). Register before Initialize; session replacement does
    // not touch them.
    RuntimeModuleGroup& PersistentModules() { return m_Persistent; }

    RenderSessionSlot& Session() { return m_Session; }

    // Optional engine allocator stack, owned here so session replacement
    // cannot release allocators that persistent services still use. The
    // root assumes exclusive ownership of the process-wide allocation hook.
    // Non-fatal: an unconfigured system falls back to the default heap.
    // Returns whether the system is configured and hooked. Call only at
    // startup, while no other thread allocates through the hook.
    bool InitializeMemorySystem(const MemorySystem::Config& config);
    // nullptr unless configured.
    MemorySystem* GetMemorySystem() { return m_pMemorySystem.get(); }

    // Recycle the per-frame arena once per frame, before module updates.
    void BeginFrame() {
        if (m_pMemorySystem) m_pMemorySystem->BeginFrame();
    }

   private:
    void ReleaseMemorySystem();
    RuntimeModuleGroup m_Persistent;
    RenderSessionSlot m_Session;
    std::unique_ptr<MemorySystem> m_pMemorySystem;
};

}  // namespace My
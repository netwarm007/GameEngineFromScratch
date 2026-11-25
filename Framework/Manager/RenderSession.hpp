#pragma once

#include <cstdint>
#include <functional>
#include <span>

#include "RuntimeModuleGroup.hpp"

namespace My {

// Identity epoch of the render session, independent of scene revision.
// Borrowed GPU references and UI caches record the generation they were
// produced under; any difference invalidates them, including during the gap
// between retirement and recreation. Starts at 0 (no session yet).
using RenderSessionGeneration = std::uint64_t;

// One replaceable render-session slot on the persistent runtime root.
// Session modules are borrowed, registered per cycle in frame order, and
// must outlive the slot. The slot never touches persistent modules: each
// Activate/Retire cycle only initializes/finalizes its own registrations.
class RenderSessionSlot {
   public:
    // Register participants for the next activation. Rejects duplicates and
    // registration while a session is active (RuntimeModuleGroup rules).
    void Register(IRuntimeModule& module) { m_Modules.Register(module); }

    // Initialize the registered session modules forward, rolling back in
    // reverse order on failure. beforeRollback lets the owner drain GPU work
    // before teardown. A successful activation publishes a new generation;
    // a failed one leaves the generation unchanged and no session active.
    int Activate(const std::function<void()>& beforeRollback = {}) {
        const int result = m_Modules.Initialize(beforeRollback);
        if (result == 0) ++m_Generation;
        return result;
    }

    // Finalize the active session in reverse order and clear registrations.
    // Retiring an active session publishes a new generation so references
    // borrowed from it invalidate immediately; repeated retirement is a
    // no-op. The owner must drain GPU work before calling this.
    void Retire() {
        if (IsActive()) ++m_Generation;
        m_Modules.Finalize();
    }

    bool IsActive() const {
        return !m_Modules.GetInitializedModules().empty();
    }

    RenderSessionGeneration Generation() const { return m_Generation; }

    // Borrowed frame participants in registration order; do not retain
    // across lifecycle changes.
    std::span<IRuntimeModule* const> GetSessionModules() const {
        return m_Modules.GetInitializedModules();
    }

   private:
    RuntimeModuleGroup m_Modules;
    RenderSessionGeneration m_Generation = 0;
};

}  // namespace My
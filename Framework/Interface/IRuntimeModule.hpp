#pragma once

#include "IApplication.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IRuntimeModule {
   public:
    IRuntimeModule() = default;
    virtual ~IRuntimeModule() = default;

    // On failure, release this module's partial initialization before returning.
    virtual int Initialize() = 0;
    virtual void Finalize() = 0;

    virtual void Tick() = 0;

    void SetAppPointer(IApplication* pApp) { m_pApp = pApp; }

   protected:
    IApplication* m_pApp = nullptr;
};
}  // namespace My

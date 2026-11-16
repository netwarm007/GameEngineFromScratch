#pragma once
#include "WindowsApplication.hpp"

#include "config.h"

#include "D3d/D3d12RHI.hpp"

namespace My {
class D3d12Application : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    D3d12RHI& GetRHI() { return m_Rhi; }

   protected:
    virtual HANDLE CreateFenceEvent() {
        return CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }

   private:
    int InitializeGraphicsResources() final;
    void WaitForGraphicsIdle() final;
    void FinalizeGraphicsResources() final;
    void onWindowResize(int new_width, int new_height) final;

   private:
    D3d12RHI m_Rhi;
};
}  // namespace My

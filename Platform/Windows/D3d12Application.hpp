#pragma once
#include "WindowsApplication.hpp"

#include "config.h"

#include "D3d/D3d12RHI.hpp"

namespace My {
class D3d12Application : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    void CreateMainWindow() override;

    D3d12RHI& GetRHI() { return m_Rhi; }

   private:
    void onWindowResize(int new_width, int new_height) override;

   private:
    D3d12RHI m_Rhi;
};
}  // namespace My

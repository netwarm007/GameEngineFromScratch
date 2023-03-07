#pragma once
#include "WindowsApplication.hpp"

#include "config.h"

#include "D2d/D2dRHI.hpp"

namespace My {
class D2dApplication : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;

    void Finalize() final;
    void CreateMainWindow() final;

    D2dRHI& GetRHI() { return m_Rhi; }

   private:
    void onWindowResize(int new_width, int new_height) final;

   private:
    D2dRHI m_Rhi;
};
}  // namespace My

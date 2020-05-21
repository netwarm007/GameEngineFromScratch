#pragma once
#include "WindowsApplication.hpp"

namespace My {
class D3d12Application : public WindowsApplication {
   public:
    using WindowsApplication::WindowsApplication;
    void Tick();
};
}  // namespace My

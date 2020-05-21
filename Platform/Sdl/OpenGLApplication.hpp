#pragma once
#include "SdlApplication.hpp"

namespace My {
class OpenGLApplication : public SdlApplication {
   public:
    using SdlApplication::SdlApplication;

    void Tick() override;

   protected:
    void CreateMainWindow() override;

   private:
    SDL_GLContext m_hContext;
};
}  // namespace My

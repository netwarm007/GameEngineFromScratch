#pragma once

#include <SDL2/SDL.h>

#include "BaseApplication.hpp"
#include "IPlatformHost.hpp"

namespace My {
class SdlApplication : public BaseApplication, public IPlatformHost {
   public:
    using BaseApplication::BaseApplication;

    int Initialize() override;
    void Finalize() override;
    // One cycle of the main loop
    void Tick() override;

    void ProcessEvents() override;

    // Backend-specific SDL ImGui binding lives in the backend application;
    // the host itself has no platform backend to bind.
    int InitializeGuiPlatform() override { return 0; }
    void FinalizeGuiPlatform() override {}

    void FinalizePlatformResources() override;

    void* GetMainWindowHandler() override { return m_pWindow; };
    void CreateMainWindow() override;

   protected:
    void logSDLError(std::ostream& os, const std::string& msg);
    void onResize(int width, int height);

   protected:
    SDL_Window* m_pWindow;

    bool m_bInDrag = false;
};
}  // namespace My

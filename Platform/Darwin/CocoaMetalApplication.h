#pragma once
#include "CocoaApplication.h"
#include "IRenderBackend.hpp"
#include "MetalBackend.h"

OBJC_CLASS(MetalView);

namespace My {
// Legacy compile-time application: the Cocoa host plus an owned Metal
// backend, with headless capture pacing. The runtime path composes the host
// and backend separately.
class CocoaMetalApplication : public CocoaApplication, public IRenderBackend {
   public:
    CocoaMetalApplication();
    explicit CocoaMetalApplication(GfxConfiguration& cfg);

    // The Metal drawable installed by the backend; nullptr before graphics
    // resources exist or when the content view is not Metal.
    MetalView* GetMetalView() { return m_Backend.GetMetalView(); }

    void CreateMainWindow() override { CocoaApplication::CreateMainWindow(); }
    int InitializeGuiPlatform() override {
        return m_Backend.InitializeGuiPlatform();
    }
    void FinalizeGuiPlatform() override { m_Backend.FinalizeGuiPlatform(); }

   private:
    int InitializeGraphicsResources() override {
        return m_Backend.InitializeGraphicsResources();
    }

    MetalBackend m_Backend;
};
}  // namespace My

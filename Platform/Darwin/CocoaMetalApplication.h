#pragma once
#include "CocoaApplication.h"
#include "IRenderBackend.hpp"

OBJC_CLASS(MetalView);

namespace My {
class CocoaMetalApplication : public CocoaApplication, public IRenderBackend {
   public:
    using CocoaApplication::CocoaApplication;

    // The Metal drawable installed by InitializeGraphicsResources; nullptr
    // before graphics resources exist or when the content view is not Metal.
    MetalView* GetMetalView();

    void Tick() override;
    void ProcessEvents() override;

   protected:
    int InitializeGraphicsResources() override;
    // The Metal RHI is owned by the graphics manager (which drains it in its
    // own Finalize); the drawable is window-owned and released with the
    // window, so the application-level hooks have no native state to drain
    // or release.
    void WaitForGraphicsIdle() override {}
    void FinalizeGraphicsResources() override {}
};
}  // namespace My

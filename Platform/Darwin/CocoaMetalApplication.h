#pragma once
#include "CocoaApplication.h"

OBJC_CLASS(MetalView);

namespace My {
class CocoaMetalApplication : public CocoaApplication {
   public:
    using CocoaApplication::CocoaApplication;

    // The Metal drawable installed by InitializeGraphicsResources; nullptr
    // before graphics resources exist or when the content view is not Metal.
    MetalView* GetMetalView();

    void Tick() override;

   protected:
    int InitializeGraphicsResources() override;
};
}  // namespace My

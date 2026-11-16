#pragma once
#include "CocoaApplication.h"

namespace My {
class CocoaMetalApplication : public CocoaApplication {
   public:
    using CocoaApplication::CocoaApplication;

    void Tick() override;

   protected:
    int InitializeGraphicsResources() override;
};
}  // namespace My

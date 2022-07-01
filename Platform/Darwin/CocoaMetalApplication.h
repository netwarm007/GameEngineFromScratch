#pragma once
#include "CocoaApplication.h"

namespace My {
class CocoaMetalApplication : public CocoaApplication {
   public:
    using CocoaApplication::CocoaApplication;

    void Tick() override;

    void Finalize() override;

    void CreateMainWindow() override;
};
}  // namespace My

#pragma once
#include "CocoaApplication.h"

namespace My {
class CocoaMetalApplication : public CocoaApplication {
   public:
    explicit CocoaMetalApplication(GfxConfiguration& config)
        : CocoaApplication(config){};

    void Tick() override;

    void CreateMainWindow() override;
};
}  // namespace My

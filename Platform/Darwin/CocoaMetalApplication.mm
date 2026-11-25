#import "CocoaMetalApplication.h"

#include "InputManager.hpp"

using namespace My;

CocoaMetalApplication::CocoaMetalApplication()
    : CocoaApplication(),
      m_Backend(*this, *this, [this] { return GetInputManager(); }) {
    SetHeadlessPacing(true);
}

CocoaMetalApplication::CocoaMetalApplication(GfxConfiguration& cfg)
    : CocoaApplication(cfg),
      m_Backend(*this, *this, [this] { return GetInputManager(); }) {
    SetHeadlessPacing(true);
}

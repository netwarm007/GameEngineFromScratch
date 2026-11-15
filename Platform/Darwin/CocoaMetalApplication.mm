#import "MetalView.h"

#include <algorithm>

#include "CocoaMetalApplication.h"

using namespace My;

void CocoaMetalApplication::CreateMainWindow() {
    CocoaApplication::CreateMainWindow();

    @autoreleasepool {
        MetalView* pView = [[MetalView new]
            initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)
                     pApp:this];

        [m_pWindow setContentView:pView];
    }
}

void CocoaMetalApplication::Tick() {
    // Physics advances a fixed step per tick. A headless `-screenshot` run without
    // `-screenshotWaitFor` is paced to 60 ticks/s so a timed capture sees the
    // same simulation time as a vsynced display, not tens of thousands of steps.
    static const bool paced = [] {
        NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
        return [defaults stringForKey:@"screenshot"] != nil &&
               [defaults stringForKey:@"screenshotWaitFor"] == nil;
    }();
    static CFTimeInterval nextTick = 0;
    if (paced) {
        const CFTimeInterval now = CACurrentMediaTime();
        if (now < nextTick) {
            [NSThread sleepForTimeInterval:nextTick - now];
        }
        nextTick = (std::max)(nextTick, now) + 1.0 / 60.0;
    }
    CocoaApplication::Tick();
}

void CocoaMetalApplication::Finalize() { CocoaApplication::Finalize(); }

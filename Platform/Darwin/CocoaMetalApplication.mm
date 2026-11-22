#import "MetalView.h"

#include <algorithm>
#include <cstdio>

#include "CocoaMetalApplication.h"

using namespace My;

MetalView* CocoaMetalApplication::GetMetalView() {
    if (!m_pWindow) return nullptr;
    NSView* content = [m_pWindow contentView];
    if (![content isKindOfClass:[MetalView class]]) return nullptr;
    return (MetalView*)content;
}

int CocoaMetalApplication::InitializeGraphicsResources() {
    if (!GetMainWindowHandler()) {
        fprintf(stderr, "Metal initialization requires a native window\n");
        return -1;
    }

    @autoreleasepool {
        MetalView* pView = [[MetalView alloc]
            initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)
                     pApp:this];

        if (!pView || !pView.device) {
            fprintf(stderr, "Metal view/device creation failed\n");
            [pView release];
            return -1;
        }
        [m_pWindow setContentView:pView];
        [pView release];
    }
    return 0;
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

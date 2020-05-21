#import "MetalView.h"

#include "CocoaMetalApplication.h"

using namespace My;

void CocoaMetalApplication::CreateMainWindow() {
    CocoaApplication::CreateMainWindow();

    MetalView* pView = [[MetalView new]
        initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)];

    [m_pWindow setContentView:pView];

    [pView release];
}

void CocoaMetalApplication::Tick() {
    CocoaApplication::Tick();
    [[m_pWindow contentView] setNeedsDisplay:YES];
}

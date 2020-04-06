#include "CocoaMetalApplication.h"
#include <climits>
#include <cstdio>
#include <cstring>

#import "MetalView.h"

using namespace My;

void CocoaMetalApplication::CreateMainWindow()
{
    CocoaApplication::CreateMainWindow();

    MetalView* pView = [MetalView new];

    [pView initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)];

    [m_pWindow setContentView:pView];
}

void CocoaMetalApplication::Tick()
{
    CocoaApplication::Tick();
    [[m_pWindow contentView] setNeedsDisplay:YES];
}


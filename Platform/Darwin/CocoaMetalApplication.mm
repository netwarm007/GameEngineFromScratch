#include <stdio.h>
#include <climits>
#include <cstring>
#include "CocoaMetalApplication.h"

#import "MetalView.h"

using namespace My;

int CocoaMetalApplication::Initialize()
{
    int result = 0;

    CocoaApplication::CreateWindow();

    MetalView* pView = [MetalView new];

    [pView initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)];

    [m_pWindow setContentView:pView];

    result = BaseApplication::Initialize();

    return result;
}

void CocoaMetalApplication::Finalize()
{
    CocoaApplication::Finalize();
}

void CocoaMetalApplication::Tick()
{
    CocoaApplication::Tick();
    [[m_pWindow contentView] setNeedsDisplay:YES];
}


#include <stdio.h>
#include <climits>
#include <cstring>
#include "CocoaOpenGLApplication.h"

#import "GLView.h"

using namespace My;

int CocoaOpenGLApplication::Initialize()
{
    int result = 0;

    result = CocoaApplication::Initialize();

    GLView* view = [[GLView alloc] initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)];

    [m_pWindow setContentView:view];

    return result;
}

void CocoaOpenGLApplication::Finalize()
{
    CocoaApplication::Finalize();
}

void CocoaOpenGLApplication::Tick()
{
    CocoaApplication::Tick();
}

void CocoaOpenGLApplication::OnDraw()
{
}


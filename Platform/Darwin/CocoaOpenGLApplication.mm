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

    if (!result) {
        NSOpenGLPixelFormatAttribute attrs[] = {
            NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
            NSOpenGLPFAColorSize,32,
            NSOpenGLPFADepthSize,16,
            NSOpenGLPFADoubleBuffer,
            NSOpenGLPFAAccelerated,
            0
        };

        GLView* view = [GLView new];
        view.pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
        [view initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)];

        if(view.pixelFormat == nil)
        {
            NSLog(@"No valid matching OpenGL Pixel Format found");
            [view release];
            return -1;
        }

        [m_pWindow setContentView:view];
    }

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


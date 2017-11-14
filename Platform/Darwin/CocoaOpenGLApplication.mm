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

        GLView* pGLView = [GLView new];
        pGLView.pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];

        if([pGLView pixelFormat] == nil)
        {
            NSLog(@"No valid matching OpenGL Pixel Format found");
            [pGLView release];
            return -1;
        }

        [pGLView initWithFrame:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)];

        [m_pWindow setContentView:pGLView];
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
    [[m_pWindow contentView] setNeedsDisplay:YES];
}


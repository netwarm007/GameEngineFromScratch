#import "GLView.h"
#import <OpenGL/gl.h>

#import "GraphicsManager.hpp"
namespace My {
    extern GraphicsManager* g_pGraphicsManager;
}

@implementation GLView

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];

    [_openGLContext makeCurrentContext];

    My::g_pGraphicsManager->Clear();
    My::g_pGraphicsManager->Draw();

    [_openGLContext flushBuffer];
}

- (instancetype)initWithFrame:(NSRect)frameRect
{
    self = [super initWithFrame:frameRect];

    _openGLContext = [[NSOpenGLContext alloc] initWithFormat:_pixelFormat shareContext:nil];

    [_openGLContext makeCurrentContext];

    [[NSNotificationCenter defaultCenter] addObserver:self
        selector:@selector(_surfaceNeedsUpdate:)
        name:NSViewGlobalFrameDidChangeNotification
        object:self];

    return self;
}

- (void)lockFocus
{
    [super lockFocus];
    if([_openGLContext view]!= self)
    {
        [_openGLContext setView:self];
    }
     [_openGLContext makeCurrentContext];

}

- (void)update
{
    [_openGLContext update];
}

- (void) _surfaceNeedsUpdate:(NSNotification*) notification
{
    [self update];

}

- (void)dealloc
{
    [_pixelFormat release];
    [_openGLContext release];

    [super dealloc];
}

@end


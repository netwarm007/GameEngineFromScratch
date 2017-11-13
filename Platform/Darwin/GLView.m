#import "GLView.h"
#import <OpenGL/gl.h>

@implementation GLView

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];

    [_openGLContext makeCurrentContext];

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    [_openGLContext flushBuffer];
}

- (instancetype)initWithFrame:(NSRect)frameRect
{
    self = [super initWithFrame:frameRect];

    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFAColorSize,32,
        NSOpenGLPFADepthSize,16,
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFAAccelerated,
        0
    };

    _pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    if(_pixelFormat == nil)
    {
        NSLog(@"No valid matching OpenGL Pixel Format found");
        return self;
    }

    _openGLContext = [[NSOpenGLContext alloc] initWithFormat:_pixelFormat shareContext:nil];


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

@end


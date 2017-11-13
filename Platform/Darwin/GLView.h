#import <Cocoa/Cocoa.h>

@interface GLView : NSView
{
    @private
    NSOpenGLContext* _openGLContext;
    NSOpenGLPixelFormat* _pixelFormat;
}

@end


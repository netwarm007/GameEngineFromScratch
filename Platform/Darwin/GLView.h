#import <Cocoa/Cocoa.h>

@interface GLView : NSView
{
    @private
    NSOpenGLContext* _openGLContext;
}

@property (strong) NSOpenGLPixelFormat* pixelFormat;

@end


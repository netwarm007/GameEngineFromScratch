#import <Cocoa/Cocoa.h>

@interface GLView : NSView
{
    @private
    NSOpenGLContext* _openGLContext;
}

@property (nonatomic, strong) NSOpenGLPixelFormat* pixelFormat;

@end


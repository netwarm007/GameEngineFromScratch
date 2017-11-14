#import <Cocoa/Cocoa.h>

@interface GLView : NSView
{
    @private
    NSOpenGLContext* _openGLContext;
    CVDisplayLinkRef _displayLink; //display link for managing rendering thread
}

@property (nonatomic, strong) NSOpenGLPixelFormat* pixelFormat;

@end


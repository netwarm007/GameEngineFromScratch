#import "MetalView.h"
#import "Metal/Metal2Renderer.h"
#include "InputManager.hpp"
#include "Metal/Metal2GraphicsManager.h"

using namespace My;

@implementation MetalView

- (instancetype)initWithCoder:(NSCoder *)coder
{
    if (self = [super initWithCoder:coder])
    {
        [self configure];
    }
    
    return self;
}

- (instancetype)initWithFrame:(CGRect)frame
{
    if (self = [super initWithFrame:frame])
    {
        [self configure];
    }
    
    return self;
}

- (instancetype)initWithFrame:(CGRect)frameRect device:(id<MTLDevice>)device
{
    if (self = [super initWithFrame:frameRect device:device])
    {
        [self configure];
    }
    
    return self;
}

- (void)configure
{
    self.device = MTLCreateSystemDefaultDevice();
    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.framebufferOnly = YES;
    self.sampleCount = 4;
    
    self.paused = YES;
    self.enableSetNeedsDisplay = YES;

    dynamic_cast<Metal2GraphicsManager*>(g_pGraphicsManager)->SetRenderer(
        [[Metal2Renderer new] initWithMetalKitView:self device:self.device] 
    );
}

- (void)mouseDown:(NSEvent *)theEvent {
    if ([theEvent type] == NSEventTypeLeftMouseDown)
    {
        g_pInputManager->LeftMouseButtonDown();
    }
}

- (void)mouseUp:(NSEvent *)theEvent {
    if ([theEvent type] == NSEventTypeLeftMouseUp)
    {
        g_pInputManager->LeftMouseButtonUp();
    }
}

- (void)mouseDragged:(NSEvent *)theEvent {
    if ([theEvent type] == NSEventTypeLeftMouseDragged)
    {
        g_pInputManager->LeftMouseDrag([theEvent deltaX], [theEvent deltaY]);
    }
}

- (void)scrollWheel:(NSEvent *)theEvent {
        g_pInputManager->LeftMouseDrag([theEvent deltaX], [theEvent deltaY]);
}

@end

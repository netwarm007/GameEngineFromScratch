#import "MetalView.h"
#import "IApplication.hpp"
#include "InputManager.hpp"
#include "Metal/Metal2GraphicsManager.h"
#include "Metal/Metal2Renderer.h"

using namespace My;

@implementation MetalView

- (instancetype)initWithCoder:(NSCoder *)coder {
    if (self = [super initWithCoder:coder]) {
        [self configure];
    }

    return self;
}

- (instancetype)initWithFrame:(CGRect)frame {
    if (self = [super initWithFrame:frame]) {
        [self configure];
    }

    return self;
}

- (instancetype)initWithFrame:(CGRect)frameRect device:(id<MTLDevice>)device {
    if (self = [super initWithFrame:frameRect device:device]) {
        [self configure];
    }

    return self;
}

- (void)configure {
    self.device = MTLCreateSystemDefaultDevice();
    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.framebufferOnly = YES;
    self.sampleCount = g_pApp->GetConfiguration().msaaSamples;

    self.paused = YES;
    self.enableSetNeedsDisplay = YES;

    dynamic_cast<Metal2GraphicsManager *>(g_pGraphicsManager)
        ->SetRenderer([[Metal2Renderer new] initWithMetalKitView:self device:self.device]);
}

- (void)drawRect:(CGRect)drawRect {
    g_pGraphicsManager->Tick();
}

- (void)mouseDown:(NSEvent *)theEvent {
    if ([theEvent type] == NSEventTypeLeftMouseDown) {
        InputManager::LeftMouseButtonDown();
    }
    else if ([theEvent type] == NSEventTypeRightMouseDown) {
        InputManager::RightMouseButtonDown();
    }
}

- (void)mouseUp:(NSEvent *)theEvent {
    if ([theEvent type] == NSEventTypeLeftMouseUp) {
        InputManager::LeftMouseButtonUp();
    }
    else if ([theEvent type] == NSEventTypeRightMouseUp) {
        InputManager::RightMouseButtonUp();
    }
}

- (void)mouseDragged:(NSEvent *)theEvent {
    if ([theEvent type] == NSEventTypeLeftMouseDragged) {
        InputManager::LeftMouseDrag([theEvent deltaX], [theEvent deltaY]);
    }
    else if ([theEvent type] == NSEventTypeRightMouseDragged) {
        InputManager::RightMouseDrag([theEvent deltaX], [theEvent deltaY]);
    }
}

- (void)scrollWheel:(NSEvent *)theEvent {
    InputManager::RightMouseDrag([theEvent deltaX], [theEvent deltaY]);
}

@end

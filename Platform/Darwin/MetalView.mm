#import "MetalView.h"

using namespace My;

@implementation MetalView {
    IInputManager* m_pInputManager;
    uint32_t m_MsaaSamples;
}

- (instancetype)initWithFrame:(CGRect)frameRect
                        input:(IInputManager*)inputManager
                  msaaSamples:(uint32_t)msaaSamples {
    m_pInputManager = inputManager;
    m_MsaaSamples = msaaSamples;

    if (self = [super initWithFrame:frameRect]) {
        [self configure];
    }

    return self;
}

- (void)configure {
    self.device = MTLCreateSystemDefaultDevice();
    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.framebufferOnly = YES;
    self.sampleCount = m_MsaaSamples;

    self.paused = YES;
    self.enableSetNeedsDisplay = YES;
}

- (void)mouseDown:(NSEvent *)theEvent {
    if (m_pInputManager) {
        if ([theEvent type] == NSEventTypeLeftMouseDown) {
            m_pInputManager->LeftMouseButtonDown();
        } else if ([theEvent type] == NSEventTypeRightMouseDown) {
            m_pInputManager->RightMouseButtonDown();
        }
    }
}

- (void)mouseUp:(NSEvent *)theEvent {
    if (m_pInputManager) {
        if ([theEvent type] == NSEventTypeLeftMouseUp) {
            m_pInputManager->LeftMouseButtonUp();
        } else if ([theEvent type] == NSEventTypeRightMouseUp) {
            m_pInputManager->RightMouseButtonUp();
        }
    }
}

- (void)mouseDragged:(NSEvent *)theEvent {
    if (m_pInputManager) {
        if ([theEvent type] == NSEventTypeLeftMouseDragged) {
            m_pInputManager->LeftMouseDrag([theEvent deltaX], [theEvent deltaY]);
        } else if ([theEvent type] == NSEventTypeRightMouseDragged) {
            m_pInputManager->RightMouseDrag([theEvent deltaX], [theEvent deltaY]);
        }
    }
}

- (void)scrollWheel:(NSEvent *)theEvent {
    if (m_pInputManager) {
        m_pInputManager->RightMouseDrag([theEvent deltaX], [theEvent deltaY]);
    }
}

@end

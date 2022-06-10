#import "MetalView.h"
#import "BaseApplication.hpp"
#include "InputManager.hpp"

using namespace My;

@implementation MetalView {
    IApplication *m_pApp;
}

- (instancetype)initWithFrame:(CGRect)frameRect pApp:(IApplication *)pApp {
    m_pApp = pApp;

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
    self.sampleCount = m_pApp->GetConfiguration().msaaSamples;

    self.paused = YES;
    self.enableSetNeedsDisplay = YES;
}

- (void)mouseDown:(NSEvent *)theEvent {
    auto pInputManager = dynamic_cast<BaseApplication *>(m_pApp)->GetInputManager();
    if (pInputManager) {
        if ([theEvent type] == NSEventTypeLeftMouseDown) {
            pInputManager->LeftMouseButtonDown();
        } else if ([theEvent type] == NSEventTypeRightMouseDown) {
            pInputManager->RightMouseButtonDown();
        }
    }
}

- (void)mouseUp:(NSEvent *)theEvent {
    auto pInputManager = dynamic_cast<BaseApplication *>(m_pApp)->GetInputManager();
    if (pInputManager) {
        if ([theEvent type] == NSEventTypeLeftMouseUp) {
            pInputManager->LeftMouseButtonUp();
        } else if ([theEvent type] == NSEventTypeRightMouseUp) {
            pInputManager->RightMouseButtonUp();
        }
    }
}

- (void)mouseDragged:(NSEvent *)theEvent {
    auto pInputManager = dynamic_cast<BaseApplication *>(m_pApp)->GetInputManager();
    if (pInputManager) {
        if ([theEvent type] == NSEventTypeLeftMouseDragged) {
            pInputManager->LeftMouseDrag([theEvent deltaX], [theEvent deltaY]);
        } else if ([theEvent type] == NSEventTypeRightMouseDragged) {
            pInputManager->RightMouseDrag([theEvent deltaX], [theEvent deltaY]);
        }
    }
}

- (void)scrollWheel:(NSEvent *)theEvent {
    auto pInputManager = dynamic_cast<BaseApplication *>(m_pApp)->GetInputManager();
    if (pInputManager) {
        pInputManager->RightMouseDrag([theEvent deltaX], [theEvent deltaY]);
    }
}

@end

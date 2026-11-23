#import "MetalBackend.h"

#import <Cocoa/Cocoa.h>

#include "IApplication.hpp"
#include "IInputManager.hpp"
#include "IPlatformHost.hpp"
#include "MetalView.h"

using namespace My;

MetalBackend::MetalBackend(IApplication& application, IPlatformHost& platform,
                           std::function<IInputManager*()> inputProvider)
    : m_Application(application),
      m_Platform(platform),
      m_InputProvider(std::move(inputProvider)) {}

MetalView* MetalBackend::GetMetalView() {
    NSWindow* window = (NSWindow*)m_Application.GetMainWindowHandler();
    if (!window) return nullptr;
    NSView* content = [window contentView];
    if (![content isKindOfClass:[MetalView class]]) return nullptr;
    return (MetalView*)content;
}

void MetalBackend::CreateMainWindow() { m_Application.CreateMainWindow(); }

int MetalBackend::InitializeGraphicsResources() {
    NSWindow* window = (NSWindow*)m_Application.GetMainWindowHandler();
    if (!window) {
        fprintf(stderr, "Metal initialization requires a native window\n");
        return -1;
    }

    @autoreleasepool {
        const auto config = m_Application.GetConfiguration();
        MetalView* pView = [[MetalView alloc]
            initWithFrame:CGRectMake(0, 0, config.screenWidth, config.screenHeight)
                    input:m_InputProvider ? m_InputProvider() : nullptr
              msaaSamples:config.msaaSamples];

        if (!pView || !pView.device) {
            fprintf(stderr, "Metal view/device creation failed\n");
            [pView release];
            return -1;
        }
        [window setContentView:pView];
        [pView release];
    }
    return 0;
}

int MetalBackend::InitializeGuiPlatform() {
    return m_Platform.InitializeGuiPlatform();
}

void MetalBackend::FinalizeGuiPlatform() { m_Platform.FinalizeGuiPlatform(); }
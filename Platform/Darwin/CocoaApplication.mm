#import <Carbon/Carbon.h>
#include <cstring>

#import "AppDelegate.h"
#include "CocoaApplication.h"
#include "InputManager.hpp"
#import "WindowDelegate.h"

#include "imgui_impl_osx.h"

using namespace My;

void* CocoaApplication::GetMainWindowHandler() { return m_pWindow; }

void CocoaApplication::CreateMainWindow() {
    [NSApplication sharedApplication];

    // Menu
    NSString* appName = [NSString stringWithFormat:@"%s", m_Config.appName];
    id menubar = [[NSMenu alloc] initWithTitle:appName];
    id appMenuItem = [NSMenuItem new];
    [menubar addItem:appMenuItem];
    [NSApp setMainMenu:menubar];
    [menubar release];

    id appMenu = [NSMenu new];
    id quitMenuItem = [[NSMenuItem alloc] initWithTitle:@"Quit"
                                                 action:@selector(terminate:)
                                          keyEquivalent:@"q"];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];
    [appMenu release];
    [appMenuItem release];

    id appDelegate = [AppDelegate new];
    [NSApp setDelegate:appDelegate];
    [appDelegate release];
    [NSApp activateIgnoringOtherApps:YES];
    [NSApp finishLaunching];

    NSInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                      NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

    NSRect contentRect =
        m_HasPreservedFrame
            ? [NSWindow contentRectForFrameRect:
                  NSMakeRect(m_PreservedX, m_PreservedY, m_PreservedW, m_PreservedH)
                                      styleMask:style]
            : NSMakeRect(0, 0, m_Config.screenWidth, m_Config.screenHeight);
    m_pWindow = [[NSWindow alloc]
        initWithContentRect:contentRect
                  styleMask:style
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [m_pWindow setTitle:appName];
    [m_pWindow makeKeyAndOrderFront:nil];
    id winDelegate = [[WindowDelegate new] initWithApp:this];
    [m_pWindow setDelegate:winDelegate];
    [winDelegate release];

}

int CocoaApplication::InitializeGuiPlatform() {
    if (!ImGui::GetCurrentContext()) return 0;
    if (!m_pWindow || ImGui::GetIO().BackendPlatformUserData) return -1;
    return ImGui_ImplOSX_Init([m_pWindow contentView]) ? 0 : -1;
}

void CocoaApplication::GetFramebufferSize(uint32_t& width, uint32_t& height) {
    @autoreleasepool {
        const NSRect contentRect = [[m_pWindow contentView] frame];
        const NSRect fbRect = [[m_pWindow contentView] convertRectToBacking:contentRect];

        width = static_cast<uint32_t>(fbRect.size.width);
        height = static_cast<uint32_t>(fbRect.size.height);
    }
}

void CocoaApplication::SetWindowTitle(const char* title) {
    if (!m_pWindow || !title) return;
    @autoreleasepool {
        [m_pWindow setTitle:[NSString stringWithUTF8String:title]];
    }
}

void CocoaApplication::FinalizeGuiPlatform() {
    if (ImGui::GetCurrentContext() && ImGui::GetIO().BackendPlatformUserData) {
        ImGui_ImplOSX_Shutdown();
    }
}

void CocoaApplication::FinalizePlatformResources() {
    if (m_pWindow) {
        const NSRect frame = [m_pWindow frame];
        m_PreservedX = frame.origin.x;
        m_PreservedY = frame.origin.y;
        m_PreservedW = frame.size.width;
        m_PreservedH = frame.size.height;
        m_HasPreservedFrame = true;
    }
    [m_pWindow release];
    m_pWindow = nil;
}

void CocoaApplication::Tick() {
    ProcessEvents();
    BaseApplication::Tick();
}

void CocoaApplication::ProcessEvents() {
    if (m_HeadlessPacing) {
        static const bool paced = [] {
            NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
            return [defaults stringForKey:@"screenshot"] != nil &&
                   [defaults stringForKey:@"screenshotWaitFor"] == nil;
        }();
        static CFTimeInterval nextTick = 0;
        if (paced) {
            const CFTimeInterval now = CACurrentMediaTime();
            if (now < nextTick) {
                [NSThread sleepForTimeInterval:nextTick - now];
            }
            nextTick = (std::max)(nextTick, now) + 1.0 / 60.0;
        }
    }
    while (NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:nil
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES]) {
        if (m_pInputManager) {
            switch ([(NSEvent*)event type]) {
                case NSEventTypeKeyUp:
                    NSLog(@"[CocoaApp] Key Up Event Received!");
                    if ([event modifierFlags] & NSEventModifierFlagNumericPad) {
                        // arrow keys
                        NSString* theArrow = [event charactersIgnoringModifiers];
                        unichar keyChar = 0;
                        if ([theArrow length] == 1) {
                            keyChar = [theArrow characterAtIndex:0];
                            if (keyChar == NSLeftArrowFunctionKey) {
                                m_pInputManager->LeftArrowKeyUp();
                                break;
                            }
                            if (keyChar == NSRightArrowFunctionKey) {
                                m_pInputManager->RightArrowKeyUp();
                                break;
                            }
                            if (keyChar == NSUpArrowFunctionKey) {
                                m_pInputManager->UpArrowKeyUp();
                                break;
                            }
                            if (keyChar == NSDownArrowFunctionKey) {
                                m_pInputManager->DownArrowKeyUp();
                                break;
                            }
                        }
                    } else {
                        switch ([event keyCode]) {
                            case kVK_ANSI_D:  // d key
                                m_pInputManager->AsciiKeyUp('d');
                                break;
                            case kVK_ANSI_R:  // r key
                                m_pInputManager->AsciiKeyUp('r');
                                break;
                            case kVK_ANSI_U:  // u key
                                m_pInputManager->AsciiKeyUp('u');
                                break;
                        }
                    }
                    break;
                case NSEventTypeKeyDown:
                    NSLog(@"[CocoaApp] Key Down Event Received! keycode=%d", [event keyCode]);
                    if ([event modifierFlags] & NSEventModifierFlagNumericPad) {
                        // arrow keys
                        NSString* theArrow = [event charactersIgnoringModifiers];
                        unichar keyChar = 0;
                        if ([theArrow length] == 1) {
                            keyChar = [theArrow characterAtIndex:0];
                            if (keyChar == NSLeftArrowFunctionKey) {
                                m_pInputManager->LeftArrowKeyDown();
                                break;
                            }
                            if (keyChar == NSRightArrowFunctionKey) {
                                m_pInputManager->RightArrowKeyDown();
                                break;
                            }
                            if (keyChar == NSUpArrowFunctionKey) {
                                m_pInputManager->UpArrowKeyDown();
                                break;
                            }
                            if (keyChar == NSDownArrowFunctionKey) {
                                m_pInputManager->DownArrowKeyDown();
                                break;
                            }
                        }
                    } else {
                        switch ([event keyCode]) {
                            case kVK_ANSI_D:  // d key
                                m_pInputManager->AsciiKeyDown('d');
                                break;
                            case kVK_ANSI_R:  // r key
                                m_pInputManager->AsciiKeyDown('r');
                                break;
                            case kVK_ANSI_U:  // u key
                                m_pInputManager->AsciiKeyDown('u');
                                break;
                        }
                    }
                    break;
                default:
                    break;
            }
        }

        [NSApp sendEvent:event];
        [NSApp updateWindows];
    }

    if (ImGui::GetCurrentContext() && ImGui::GetIO().BackendPlatformUserData) {
        ImGui_ImplOSX_NewFrame([m_pWindow contentView]);
    }
}

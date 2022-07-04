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

    m_pWindow = [[NSWindow alloc]
        initWithContentRect:CGRectMake(0, 0, m_Config.screenWidth, m_Config.screenHeight)
                  styleMask:style
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [m_pWindow setTitle:appName];
    [m_pWindow makeKeyAndOrderFront:nil];
    id winDelegate = [WindowDelegate new];
    [m_pWindow setDelegate:winDelegate];
    [winDelegate release];
}

void CocoaApplication::GetFramebufferSize(int& width, int& height) {
    @autoreleasepool {
        const NSRect contentRect = [[m_pWindow contentView] frame];
        const NSRect fbRect = [[m_pWindow contentView] convertRectToBacking:contentRect];

        width = static_cast<int>(fbRect.size.width);
        height = static_cast<int>(fbRect.size.height);
    }
}

void CocoaApplication::Finalize() {
    [m_pWindow release];
    BaseApplication::Finalize();
}

void CocoaApplication::Tick() {
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

    if (ImGui::GetCurrentContext()) {
        ImGui_ImplOSX_NewFrame([m_pWindow contentView]);
    }
    BaseApplication::Tick();
}

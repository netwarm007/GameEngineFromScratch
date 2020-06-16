#include "CocoaApplication.h"
#import <Carbon/Carbon.h>
#include <cstring>
#import "AppDelegate.h"
#include "InputManager.hpp"
#import "WindowDelegate.h"
#include "imgui/examples/imgui_impl_osx.h"

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

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    [[maybe_unused]] ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplOSX_Init();

    ImGui::StyleColorsDark();
}

void CocoaApplication::Finalize() {
    ImGui_ImplOSX_Shutdown();

    [m_pWindow release];
    BaseApplication::Finalize();
}

void CocoaApplication::Tick() {
    while (NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:nil
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES]) {

        ImGui_ImplOSX_HandleEvent(event, [m_pWindow contentView]);

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
                            g_pInputManager->LeftArrowKeyUp();
                            break;
                        }
                        if (keyChar == NSRightArrowFunctionKey) {
                            g_pInputManager->RightArrowKeyUp();
                            break;
                        }
                        if (keyChar == NSUpArrowFunctionKey) {
                            g_pInputManager->UpArrowKeyUp();
                            break;
                        }
                        if (keyChar == NSDownArrowFunctionKey) {
                            g_pInputManager->DownArrowKeyUp();
                            break;
                        }
                    }
                } else {
                    switch ([event keyCode]) {
                        case kVK_ANSI_D:  // d key
                            InputManager::AsciiKeyUp('d');
                            break;
                        case kVK_ANSI_R:  // r key
                            InputManager::AsciiKeyUp('r');
                            break;
                        case kVK_ANSI_U:  // u key
                            InputManager::AsciiKeyUp('u');
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
                            g_pInputManager->LeftArrowKeyDown();
                            break;
                        }
                        if (keyChar == NSRightArrowFunctionKey) {
                            g_pInputManager->RightArrowKeyDown();
                            break;
                        }
                        if (keyChar == NSUpArrowFunctionKey) {
                            g_pInputManager->UpArrowKeyDown();
                            break;
                        }
                        if (keyChar == NSDownArrowFunctionKey) {
                            g_pInputManager->DownArrowKeyDown();
                            break;
                        }
                    }
                } else {
                    switch ([event keyCode]) {
                        case kVK_ANSI_D:  // d key
                            My::InputManager::AsciiKeyDown('d');
                            break;
                        case kVK_ANSI_R:  // r key
                            My::InputManager::AsciiKeyDown('r');
                            break;
                        case kVK_ANSI_U:  // u key
                            My::InputManager::AsciiKeyDown('u');
                            break;
                    }
                }
                break;
            default:
                break;
        }
        [NSApp sendEvent:event];
        [NSApp updateWindows];
    }
}

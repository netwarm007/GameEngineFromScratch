#include <string.h>
#include "CocoaApplication.hpp"
#include "MemoryManager.hpp"
#include "GraphicsManager.hpp"

#import <AppDelegate.h>
#import <WindowDelegate.h>
#import <GLView.h>

using namespace My;

int CocoaApplication::Initialize()
{
    int result = 0;

    [NSApplication  sharedApplication];

    // Menu
    NSString* appName = [NSString stringWithFormat:@"%s", m_Config.appName];
    id menubar = [[NSMenu alloc] initWithTitle:appName];
    id appMenuItem = [NSMenuItem new];
    [menubar addItem: appMenuItem];
    [NSApp setMainMenu:menubar];

    id appMenu = [NSMenu new];
    id quitMenuItem = [[NSMenuItem alloc] initWithTitle:@"Quit"
        action:@selector(terminate:)
        keyEquivalent:@"q"];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];

    id appDelegate = [AppDelegate new];
    [NSApp setDelegate: appDelegate];
    [NSApp activateIgnoringOtherApps:YES];
    [NSApp finishLaunching];

    NSInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                      NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

    GLView* view = [[GLView alloc] initWithFrame:CGRectMake(0, 0, 800, 600)];

    m_pWindow = [[NSWindow alloc] initWithContentRect:CGRectMake(0, 0, 800, 600) styleMask:style backing:NSBackingStoreBuffered defer:NO];
    [m_pWindow setTitle:appName];
    [m_pWindow setContentView:view];
    [m_pWindow makeKeyAndOrderFront:nil];
    id winDelegate = [WindowDelegate new];
    [m_pWindow setDelegate:winDelegate];

    return result;
}

void CocoaApplication::Finalize()
{
    [m_pWindow release];
}

void CocoaApplication::Tick()
{
    NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny
    untilDate:nil
    inMode:NSDefaultRunLoopMode
    dequeue:YES];

    switch([(NSEvent *)event type])
    {
        case NSEventTypeKeyDown:
            NSLog(@"Key Down Event Received!");
            break;
        default:
            break;
    }
    [NSApp sendEvent:event];
    [NSApp updateWindows];
    [event release];
}



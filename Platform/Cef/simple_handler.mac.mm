// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

// modified by Chen, Wenli on 2018/12/25 to integrate to GameEngineFromScratch

#include "simple_handler.hpp"

#import <Cocoa/Cocoa.h>

#include "cef_browser.h"

using namespace My;

void SimpleHandler::PlatformTitleChange(const CefRefPtr<CefBrowser>& browser,
                                        const CefString& title) {
    NSView* view = (NSView*)browser->GetHost()->GetWindowHandle();
    NSWindow* window = [view window];
    std::string titleStr(title);
    NSString* str = [NSString stringWithUTF8String:titleStr.c_str()];
    if (str != nil) {
        [window setTitle:str];
    }
}
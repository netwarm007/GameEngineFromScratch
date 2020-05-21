// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

// Modified by Chen Wenli @ 2018/12/21 for integration purpose

#include "CefApplication.hpp"

#include <string>

#include "cef_browser.h"
#include "simple_handler.hpp"
#include "views/cef_browser_view.h"
#include "views/cef_window.h"
#include "wrapper/cef_helpers.h"

namespace My {
// When using the Views framework this object provides the delegate
// implementation for the CefWindow that hosts the Views-based browser.
class CefAppWindowDelegate : public CefWindowDelegate {
   public:
    explicit CefAppWindowDelegate(const CefRefPtr<CefBrowserView>& browser_view)
        : browser_view_(browser_view) {}

    void OnWindowCreated(CefRefPtr<CefWindow> window) override {
        // Add the browser view and show the window.
        window->AddChildView(browser_view_);
        window->Show();

        // Give keyboard focus to the browser view.
        browser_view_->RequestFocus();
    }

    void OnWindowDestroyed(CefRefPtr<CefWindow> window) override {
        browser_view_ = nullptr;
    }

    bool CanClose(CefRefPtr<CefWindow> window) override {
        // Allow the window to close if the browser says it's OK.
        CefRefPtr<CefBrowser> browser = browser_view_->GetBrowser();
        if (browser) {
            return browser->GetHost()->TryCloseBrowser();
        }
        return true;
    }

   private:
    CefRefPtr<CefBrowserView> browser_view_;

    IMPLEMENT_REFCOUNTING(CefAppWindowDelegate);
    DISALLOW_COPY_AND_ASSIGN(CefAppWindowDelegate);
};
}  // namespace My

using namespace My;

void CefApplication::OnContextInitialized() {
    CEF_REQUIRE_UI_THREAD();

    CefRefPtr<CefCommandLine> command_line =
        CefCommandLine::GetGlobalCommandLine();

#if defined(OS_WIN) || defined(OS_LINUX)
    // Create the browser using the Views framework if "--use-views" is
    // specified via the command-line. Otherwise, create the browser using the
    // native platform framework. The Views framework is currently only
    // supported on Windows and Linux.
    const bool use_views = command_line->HasSwitch("use-views");
#else
    const bool use_views = false;
#endif

    // SimpleHandler implements browser-level callbacks.
    CefRefPtr<SimpleHandler> handler(new SimpleHandler(use_views));

    // Specify CEF browser settings here.
    CefBrowserSettings browser_settings;

    std::string url;

    // Check if a "--url=" value was provided via the command-line. If so, use
    // that instead of the default URL.
    url = command_line->GetSwitchValue("url");
    if (url.empty()) {
        url = "http://desktop.chenwenli.com/Viewer.html";
    }

    if (use_views) {
        // Create the BrowserView.
        CefRefPtr<CefBrowserView> browser_view =
            CefBrowserView::CreateBrowserView(handler, url, browser_settings,
                                              nullptr, nullptr, nullptr);

        // Create the Window. It will show itself after creation.
        CefWindow::CreateTopLevelWindow(new CefAppWindowDelegate(browser_view));
    } else {
        // Information used when creating the native window.
        CefWindowInfo window_info;

        window_info.width = m_Config.screenWidth;
        window_info.height = m_Config.screenHeight;

#if defined(OS_WIN)
        // On Windows we need to specify certain flags that will be passed to
        // CreateWindowEx().
        window_info.SetAsPopup(NULL, m_Config.appName);
#endif

        // Create the first browser window.
        CefBrowserHost::CreateBrowser(window_info, handler, url,
                                      browser_settings, nullptr, nullptr);
    }
}

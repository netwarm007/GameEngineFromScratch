// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

// Modified by Chen Wenli @ 2018/12/21 for integration purpose

#pragma once
#include <list>

#include "cef_client.h"

namespace My {
class SimpleHandler : public CefClient,
                      public CefDisplayHandler,
                      public CefLifeSpanHandler,
                      public CefLoadHandler {
   public:
    explicit SimpleHandler(bool use_views);
    ~SimpleHandler() override;

    // Provide access to the single global instance of this object.
    static SimpleHandler* GetInstance();

    // CefClient methods:
    CefRefPtr<CefDisplayHandler> GetDisplayHandler() override { return this; }
    CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override { return this; }
    CefRefPtr<CefLoadHandler> GetLoadHandler() override { return this; }

    // CefDisplayHandler methods:
    void OnTitleChange(CefRefPtr<CefBrowser> browser,
                       const CefString& title) override;

    // CefLifeSpanHandler methods:
    void OnAfterCreated(CefRefPtr<CefBrowser> browser) override;
    bool DoClose(CefRefPtr<CefBrowser> browser) override;
    void OnBeforeClose(CefRefPtr<CefBrowser> browser) override;

    // CefLoadHandler methods:
    void OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                     ErrorCode errorCode, const CefString& errorText,
                     const CefString& failedUrl) override;

    // Request that all existing browser windows close.
    void CloseAllBrowsers(bool force_close);

    bool IsClosing() const { return is_closing_; }

   private:
    // Platform-specific implementation.
    static void PlatformTitleChange(const CefRefPtr<CefBrowser>& browser,
                                    const CefString& title);

    // True if the application is using the Views framework.
    const bool use_views_;

    // List of existing browser windows. Only accessed on the CEF UI thread.
    using BrowserList = std::list<CefRefPtr<CefBrowser> >;
    BrowserList browser_list_;

    bool is_closing_;

    // Include the default reference counting implementation.
    IMPLEMENT_REFCOUNTING(SimpleHandler);
};
}  // namespace My